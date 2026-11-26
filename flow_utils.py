"""Shared helpers for the Amazon Bedrock Flows modules.

Module 01 shows every Flows API call by hand. From module 02 on, the modules
focus on one node type each and use these helpers for the repeated plumbing:
the IAM role, the create -> prepare -> version -> alias lifecycle, invoking a
flow, and cleanup.

Nothing here is hard-coded to an account or region: both come from your
current AWS login and configuration.
"""
import io
import json
import re
import time
import uuid
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROLE_NAME = "BedrockFlowsSampleRole"
LAMBDA_ROLE_NAME = "BedrockFlowsSampleLambdaRole"
LAMBDA_PREFIX = "BedrockFlowsSample"  # the flow role may only invoke Lambda functions with this name prefix
AGENT_ROLE_NAME = "BedrockFlowsSampleAgentRole"
VARIABLE_PATTERN = re.compile(r"\$\$[A-Za-z][A-Za-z0-9_]*")


# ---------------------------------------------------------------- session ---
def get_session_info():
    """Return (session, region, account_id) from the current AWS login."""
    session = boto3.Session()
    region = session.region_name
    if not region:
        raise SystemExit(
            "No AWS region found. Set AWS_REGION or run: aws configure set region <region>"
        )
    account = session.client("sts").get_caller_identity()["Account"]
    return session, region, account


def inference_profile_arn(region, account_id, model):
    """ARN of the cross-region inference profile for `model` (for example "amazon.nova-2-lite-v1:0") in this region.

    Profile prefixes differ by region (us., eu., jp., au., apac., ...), and some regions only offer the
    global. profile, so this looks the profile up instead of guessing. It prefers a geographic profile,
    which keeps requests in your geography, and falls back to global.
    """
    bedrock = boto3.client("bedrock", region_name=region)
    ids = [p["inferenceProfileId"]
           for page in bedrock.get_paginator("list_inference_profiles").paginate(typeEquals="SYSTEM_DEFINED")
           for p in page["inferenceProfileSummaries"]]
    matches = [i for i in ids if i.split(".", 1)[-1] == model]
    geographic = [i for i in matches if not i.startswith("global.")]
    chosen = (geographic or matches or [None])[0]
    if not chosen:
        raise SystemExit(f"No inference profile for {model} in {region}. Pass --model-id.")
    return f"arn:aws:bedrock:{region}:{account_id}:inference-profile/{chosen}"


def default_model_id(region, account_id):
    """ARN of the Amazon Nova 2 Lite inference profile for this region.

    The Flows docs ask for the inference profile ARN in a prompt node's modelId. The bare profile ID
    also runs, but the console's prompt node editor reports it as an invalid model identifier.
    """
    return inference_profile_arn(region, account_id, "amazon.nova-2-lite-v1:0")


def default_agent_model_id(region, account_id):
    """ARN of the Claude Haiku 4.5 inference profile for this region, the model module 05's agents use.

    With the default Agents Classic orchestration prompt, Nova 2 Lite sometimes ends a turn without a tool
    call or an answer when it wants to search again, and the agent replies "Sorry, I am unable to assist you
    with this request." Newer Claude models (for example Haiku 5.5) reject the thinking settings Agents
    Classic sends.
    """
    return inference_profile_arn(region, account_id, "anthropic.claude-haiku-4-5-20251001-v1:0")


# -------------------------------------------------------------------- IAM ---
def build_trust_policy(region, account_id):
    """Only Bedrock flows in this account and region may assume the role (confused-deputy protection)."""
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "FlowsTrustBedrock",
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:flow/*"},
            },
        }],
    }


def build_permissions_policy(region, account_id):
    """Runtime permissions for flow nodes, scoped to this account and region where the ARN allows.

    Control-plane actions (CreateFlow, DeleteFlow, ...) are not needed by the
    execution role. Your own credentials call them.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeModels",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
                    f"arn:aws:bedrock:{region}:{account_id}:provisioned-model/*",
                    f"arn:aws:bedrock:{region}:{account_id}:custom-model/*",
                ],
            },
            {
                # A Knowledge Base node calls RetrieveAndGenerate, which looks up the inference profile
                # and its models before invoking. Prompt nodes don't need these.
                "Sid": "ResolveInferenceProfiles",
                "Effect": "Allow",
                "Action": ["bedrock:GetInferenceProfile", "bedrock:GetFoundationModel"],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
                ],
            },
            {
                "Sid": "PromptNodes",
                "Effect": "Allow",
                "Action": ["bedrock:GetPrompt", "bedrock:RenderPrompt"],
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:prompt/*",
            },
            {
                "Sid": "KnowledgeBaseNodes",
                "Effect": "Allow",
                "Action": ["bedrock:GetKnowledgeBase", "bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*",
            },
            {
                "Sid": "AgentNodes",
                "Effect": "Allow",
                "Action": ["bedrock:GetAgent", "bedrock:InvokeAgent"],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account_id}:agent/*",
                    f"arn:aws:bedrock:{region}:{account_id}:agent-alias/*",
                ],
            },
            {
                "Sid": "Guardrails",
                "Effect": "Allow",
                "Action": ["bedrock:GetGuardrail", "bedrock:ApplyGuardrail"],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account_id}:guardrail/*",
                    # cross-region guardrail profiles (e.g. us.guardrail.v1:0) can live in other regions
                    f"arn:aws:bedrock:*:{account_id}:guardrail-profile/*",
                ],
            },
            {
                "Sid": "LambdaNodes",
                "Effect": "Allow",
                "Action": ["lambda:InvokeFunction"],
                "Resource": f"arn:aws:lambda:{region}:{account_id}:function:{LAMBDA_PREFIX}*",
            },
        ],
    }


def ensure_role(iam, region, account_id, name=ROLE_NAME):
    """Create the flows execution role if needed and make sure its policies match this region.

    Safe to run repeatedly. Returns the role ARN.
    """
    trust = json.dumps(build_trust_policy(region, account_id))
    created = False
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust)
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=trust,
            Description="Execution role for the Bedrock Flows sample modules",
        )["Role"]["Arn"]
        created = True
    iam.put_role_policy(
        RoleName=name,
        PolicyName="BedrockFlowsPolicy",
        PolicyDocument=json.dumps(build_permissions_policy(region, account_id)),
    )
    print(f"{'Created' if created else 'Using'} role: {arn}")
    if created:
        time.sleep(10)  # IAM is eventually consistent; give the new role time to propagate
    return arn


def delete_role(iam, name=ROLE_NAME):
    """Delete the sample execution role (inline policy first)."""
    try:
        iam.delete_role_policy(RoleName=name, PolicyName="BedrockFlowsPolicy")
        iam.delete_role(RoleName=name)
        print(f"Deleted role: {name}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role {name} does not exist")


# ---------------------------------------------------------------- Lambda ---
def ensure_lambda_role(iam, name=LAMBDA_ROLE_NAME):
    """Execution role for the sample Lambda functions: CloudWatch Logs only. Returns the role ARN."""
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    arn = iam.create_role(
        RoleName=name,
        AssumeRolePolicyDocument=trust,
        Description="Execution role for the Bedrock Flows sample Lambda functions",
    )["Role"]["Arn"]
    iam.attach_role_policy(
        RoleName=name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    )
    print(f"Created Lambda role: {arn}")
    return arn


def deploy_lambda(lambda_client, iam, name, source_path, timeout=30):
    """Create (or update) a single-file Python Lambda function and wait until it is ready. Returns its ARN.

    `name` must start with LAMBDA_PREFIX so the flow execution role is allowed to invoke it.
    The handler is `<file name>.lambda_handler`.
    """
    if not name.startswith(LAMBDA_PREFIX):
        raise ValueError(f"Lambda name must start with {LAMBDA_PREFIX}")
    source_path = Path(source_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(source_path, source_path.name)
    code = buffer.getvalue()

    try:
        arn = lambda_client.get_function(FunctionName=name)["Configuration"]["FunctionArn"]
        lambda_client.update_function_code(FunctionName=name, ZipFile=code)
        lambda_client.get_waiter("function_updated_v2").wait(FunctionName=name)
        print(f"Updated Lambda function: {name}")
        return arn
    except lambda_client.exceptions.ResourceNotFoundException:
        pass

    role_arn = ensure_lambda_role(iam)
    for attempt in range(1, 7):
        try:
            arn = lambda_client.create_function(
                FunctionName=name,
                Runtime="python3.13",
                Role=role_arn,
                Handler=f"{source_path.stem}.lambda_handler",
                Code={"ZipFile": code},
                Timeout=timeout,
                Description="Bedrock Flows sample",
            )["FunctionArn"]
            break
        except lambda_client.exceptions.InvalidParameterValueException as e:
            if "role" not in str(e).lower() or attempt == 6:
                raise
            print(f"Lambda role not usable yet, retrying ({attempt}/5)...")
            time.sleep(5)  # a brand-new IAM role takes a few seconds before Lambda can assume it
    lambda_client.get_waiter("function_active_v2").wait(FunctionName=name)
    print(f"Created Lambda function: {name}")
    return arn


def delete_lambda(lambda_client, name):
    """Delete a sample Lambda function."""
    try:
        lambda_client.delete_function(FunctionName=name)
        print(f"Deleted Lambda function: {name}")
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"Lambda function {name} does not exist")


def delete_lambda_role(iam, name=LAMBDA_ROLE_NAME):
    """Delete the sample Lambda execution role (detach its managed policy first)."""
    try:
        iam.detach_role_policy(
            RoleName=name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
        iam.delete_role(RoleName=name)
        print(f"Deleted role: {name}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role {name} does not exist")


# ---------------------------------------------------------------- agents ---
# Amazon Bedrock Agents (now "Agents Classic") is in maintenance mode and closed to new customers.
# Accounts that already use it can still create agents. New projects should look at Amazon Bedrock AgentCore.
def ensure_agent_role(iam, region, account_id, name=AGENT_ROLE_NAME):
    """Service role the sample agents run under: invoke models through inference profiles only. Returns its ARN.

    Follows "Create a service role for Amazon Bedrock Agents" in the Bedrock User Guide, including the
    extra actions an agent needs when its model is an inference profile.
    """
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:agent/*"},
            },
        }],
    })
    permissions = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AgentModelInvocationPermissions",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:GetInferenceProfile",
                "bedrock:GetFoundationModel",
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/*",  # a cross-region profile routes to several regions
                f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
            ],
        }],
    })
    created = False
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust)
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=trust,
            Description="Service role for the Bedrock Flows sample agents",
        )["Role"]["Arn"]
        created = True
    iam.put_role_policy(RoleName=name, PolicyName="BedrockFlowsAgentPolicy", PolicyDocument=permissions)
    print(f"{'Created' if created else 'Using'} agent role: {arn}")
    if created:
        time.sleep(10)  # IAM is eventually consistent
    return arn


def _wait_for(get_status, wanted, label, timeout=180):
    deadline = time.time() + timeout
    while True:
        status = get_status()
        if status == wanted:
            return
        if status in ("FAILED", "DELETING"):
            raise RuntimeError(f"{label} is {status}")
        if time.time() > deadline:
            raise TimeoutError(f"{label} not {wanted} after {timeout}s (status: {status})")
        time.sleep(3)


def create_agent(agent, name, instruction, model_id, role_arn, alias_name="live", action_groups=(), before_prepare=None):
    """Create an agent, add its action groups (tools), prepare it, and give it an alias.

    Returns (agent_id, alias_id, alias_arn). Each item in `action_groups` is the keyword arguments for
    create_agent_action_group (without agentId/agentVersion). `before_prepare(agent_id)` runs once the
    agent exists, for example to let it invoke a Lambda function.

    A flow's agent node needs an alias ARN. The console's built-in test alias (TSTALIASID) does not work there.
    """
    for attempt in range(1, 7):
        try:
            agent_id = agent.create_agent(
                agentName=name,
                instruction=instruction,
                foundationModel=model_id,
                agentResourceRoleArn=role_arn,
                idleSessionTTLInSeconds=600,
                description="Bedrock Flows sample agent",
            )["agent"]["agentId"]
            break
        except ClientError as e:
            message = e.response["Error"].get("Message", "").lower()
            if e.response["Error"]["Code"] == "ValidationException" and "role" in message and attempt < 6:
                print(f"Agent role not usable yet, retrying ({attempt}/5)...")
                time.sleep(5)
                continue
            raise
    get_status = lambda: agent.get_agent(agentId=agent_id)["agent"]["agentStatus"]  # noqa: E731
    _wait_for(get_status, "NOT_PREPARED", f"Agent {name}")  # CREATING -> NOT_PREPARED
    if before_prepare:
        before_prepare(agent_id)
    for group in action_groups:
        agent.create_agent_action_group(agentId=agent_id, agentVersion="DRAFT", **group)
    agent.prepare_agent(agentId=agent_id)
    _wait_for(get_status, "PREPARED", f"Agent {name}")
    alias = agent.create_agent_alias(agentId=agent_id, agentAliasName=alias_name)["agentAlias"]
    alias_id = alias["agentAliasId"]
    _wait_for(lambda: agent.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasStatus"],
              "PREPARED", f"Alias of agent {name}")
    print(f"Created agent {name}: {agent_id} (alias {alias_id})")
    return agent_id, alias_id, alias["agentAliasArn"]


def allow_agent_to_invoke_lambda(lambda_client, function_name, region, account_id, agent_id):
    """Resource-based policy on the function so this one agent can call it as an action group tool.

    Agents need this on the Lambda function itself ("Resource-based policy to allow Amazon Bedrock to
    invoke an action group Lambda function" in the Bedrock User Guide).
    """
    lambda_client.add_permission(
        FunctionName=function_name,
        StatementId=f"AllowBedrockAgent-{agent_id}",
        Action="lambda:InvokeFunction",
        Principal="bedrock.amazonaws.com",
        SourceAccount=account_id,
        SourceArn=f"arn:aws:bedrock:{region}:{account_id}:agent/{agent_id}",
    )


def invoke_agent(runtime, agent_id, alias_id, text, show_steps=True):
    """Invoke an agent directly and return its answer. With show_steps, print each tool call it makes.

    The trace is how you see the agent decide: its reasoning, which tool it calls with what input,
    and what the tool returned.
    """
    response = runtime.invoke_agent(
        agentId=agent_id, agentAliasId=alias_id, sessionId=str(uuid.uuid4()),
        inputText=text, enableTrace=show_steps,
    )
    answer, tool_calls = [], 0
    for event in response["completion"]:
        if "chunk" in event:
            answer.append(event["chunk"]["bytes"].decode())
        orchestration = event.get("trace", {}).get("trace", {}).get("orchestrationTrace", {})
        if "rationale" in orchestration:
            print(f"  [thinks] {orchestration['rationale'].get('text', '').strip()[:300]}")
        call = orchestration.get("invocationInput", {}).get("actionGroupInvocationInput")
        if call:
            tool_calls += 1
            args = {p["name"]: p["value"] for p in call.get("parameters", [])}
            print(f"  [calls]  {call.get('function')}({args})")
        result = orchestration.get("observation", {}).get("actionGroupInvocationOutput")
        if result:
            print(f"  [gets]   {len(result.get('text', ''))} characters of results")
    if show_steps:
        print(f"  ({tool_calls} tool call{'s' if tool_calls != 1 else ''})")
    return "".join(answer)


def delete_agent(agent, agent_id, alias_id=None):
    """Delete an agent's alias, then the agent (which also deletes its versions)."""
    try:
        if alias_id:
            agent.delete_agent_alias(agentId=agent_id, agentAliasId=alias_id)
            print(f"Deleted agent alias: {alias_id}")
        agent.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)
        print(f"Deleted agent: {agent_id}")
    except Exception as e:
        print(f"Could not delete agent {agent_id}: {e}")


def delete_agent_role(iam, name=AGENT_ROLE_NAME):
    """Delete the sample agent service role (inline policy first)."""
    try:
        iam.delete_role_policy(RoleName=name, PolicyName="BedrockFlowsAgentPolicy")
        iam.delete_role(RoleName=name)
        print(f"Deleted role: {name}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role {name} does not exist")


# -------------------------------------------------------------- templates ---
def load_template(path, values):
    """Load a flow template JSON and substitute its $$VARIABLES.

    Substitution happens on the parsed JSON (whole variable names only), so a
    value can never break the JSON or collide with a longer variable name.
    Returns the template dict with `definition` filled in.
    """
    template = json.loads(Path(path).read_text())
    found = set()
    _collect(template["definition"], found)
    missing = sorted(found - set(values))
    if missing:
        raise SystemExit(f"Missing values for template variables: {', '.join(missing)}")
    template["definition"] = _substitute(template["definition"], values)
    return template


def _collect(node, found):
    if isinstance(node, str):
        found.update(VARIABLE_PATTERN.findall(node))
    elif isinstance(node, dict):
        for value in node.values():
            _collect(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect(value, found)


def _substitute(node, values):
    if isinstance(node, str):
        return VARIABLE_PATTERN.sub(lambda m: values.get(m.group(0), m.group(0)), node)
    if isinstance(node, dict):
        return {k: _substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    return node


# ------------------------------------------------------- knowledge bases ---
KB_ROLE_NAME = "BedrockFlowsSampleKnowledgeBaseRole"
KB_NAME = "BedrockFlowsSampleKnowledgeBase"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS = 1024  # must match the vector index dimension


def sample_kb_resources(region, account_id):
    """Names of everything create_sample_knowledge_base makes. S3 bucket names are global, so they include the account."""
    return {
        "docs_bucket": f"bedrock-flows-sample-kb-docs-{account_id}-{region}",
        "vector_bucket": f"bedrock-flows-sample-vectors-{account_id}",
        "index": "sample-docs",
    }


def ensure_kb_role(iam, region, account_id, names, name=KB_ROLE_NAME):
    """Service role the knowledge base uses to read the documents, embed them, and use the vector index.

    Follows "Create a service role for Amazon Bedrock Knowledge Bases" in the Bedrock User Guide:
    trust, embedding model access, the S3 data source, and the S3 Vectors index.
    """
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*"},
            },
        }],
    })
    bucket = names["docs_bucket"]
    permissions = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ListModels", "Effect": "Allow",
             "Action": ["bedrock:ListFoundationModels", "bedrock:ListCustomModels"], "Resource": "*"},
            {"Sid": "EmbeddingModel", "Effect": "Allow", "Action": ["bedrock:InvokeModel"],
             "Resource": f"arn:aws:bedrock:{region}::foundation-model/{EMBEDDING_MODEL_ID}"},
            {"Sid": "S3ListBucketStatement", "Effect": "Allow", "Action": ["s3:ListBucket"],
             "Resource": f"arn:aws:s3:::{bucket}",
             "Condition": {"StringEquals": {"aws:ResourceAccount": account_id}}},
            {"Sid": "S3GetObjectStatement", "Effect": "Allow", "Action": ["s3:GetObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*",
             "Condition": {"StringEquals": {"aws:ResourceAccount": account_id}}},
            {"Sid": "S3VectorBucketReadAndWritePermission", "Effect": "Allow",
             "Action": ["s3vectors:PutVectors", "s3vectors:GetVectors", "s3vectors:DeleteVectors",
                        "s3vectors:QueryVectors", "s3vectors:GetIndex"],
             "Resource": f"arn:aws:s3vectors:{region}:{account_id}:bucket/{names['vector_bucket']}/index/{names['index']}"},
        ],
    })
    created = False
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust)
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust,
                              Description="Service role for the Bedrock Flows sample knowledge base")["Role"]["Arn"]
        created = True
    iam.put_role_policy(RoleName=name, PolicyName="BedrockFlowsKnowledgeBasePolicy", PolicyDocument=permissions)
    print(f"{'Created' if created else 'Using'} knowledge base role: {arn}")
    if created:
        time.sleep(10)  # IAM is eventually consistent
    return arn


def create_sample_knowledge_base(session, region, account_id, docs_dir, state):
    """Build a knowledge base on Amazon S3 Vectors from the files in docs_dir. Returns the knowledge base ID.

    Steps: an S3 bucket for the documents, an S3 vector bucket and index for the embeddings, the service
    role, the knowledge base, an S3 data source, and an ingestion job that chunks and embeds the documents.
    Every resource is recorded in `state` as it is created, so delete_sample_knowledge_base(state) can
    clean up even after a failure partway through.
    """
    s3 = session.client("s3")
    s3vectors = session.client("s3vectors")
    agent = session.client("bedrock-agent")
    iam = session.client("iam")
    names = sample_kb_resources(region, account_id)
    state.update(region=region, account_id=account_id, **names)

    # 1. The documents, in a private S3 bucket (new buckets block public access by default).
    bucket = names["docs_bucket"]
    try:
        args = {"Bucket": bucket}
        if region != "us-east-1":
            args["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**args)
        print(f"Created S3 bucket: {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Using S3 bucket: {bucket}")
    state["docs_bucket_created"] = True
    for path in sorted(Path(docs_dir).glob("*.md")):
        s3.upload_file(str(path), bucket, path.name)
        print(f"  uploaded {path.name}")

    # 2. The vector store: an S3 vector bucket and an index sized for the embedding model.
    try:
        s3vectors.create_vector_bucket(vectorBucketName=names["vector_bucket"])
        print(f"Created vector bucket: {names['vector_bucket']}")
    except s3vectors.exceptions.ConflictException:
        print(f"Using vector bucket: {names['vector_bucket']}")
    state["vector_bucket_created"] = True
    try:
        s3vectors.create_index(
            vectorBucketName=names["vector_bucket"], indexName=names["index"],
            dataType="float32", dimension=EMBEDDING_DIMENSIONS, distanceMetric="cosine",
            # Bedrock stores each chunk's text in AMAZON_BEDROCK_TEXT. Non-filterable keys may be larger.
            metadataConfiguration={"nonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT"]},
        )
        print(f"Created vector index: {names['index']}")
    except s3vectors.exceptions.ConflictException:
        print(f"Using vector index: {names['index']}")
    state["index_created"] = True
    index_arn = s3vectors.get_index(vectorBucketName=names["vector_bucket"], indexName=names["index"])["index"]["indexArn"]

    # 3. The knowledge base and its service role.
    role_arn = ensure_kb_role(iam, region, account_id, names)
    for attempt in range(1, 7):
        try:
            kb_id = agent.create_knowledge_base(
                name=KB_NAME,
                description="Bedrock Flows sample knowledge base on Amazon S3 Vectors",
                roleArn=role_arn,
                knowledgeBaseConfiguration={
                    "type": "VECTOR",
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": f"arn:aws:bedrock:{region}::foundation-model/{EMBEDDING_MODEL_ID}",
                        "embeddingModelConfiguration": {"bedrockEmbeddingModelConfiguration": {
                            "dimensions": EMBEDDING_DIMENSIONS, "embeddingDataType": "FLOAT32"}},
                    },
                },
                storageConfiguration={"type": "S3_VECTORS", "s3VectorsConfiguration": {"indexArn": index_arn}},
            )["knowledgeBase"]["knowledgeBaseId"]
            break
        except ClientError as e:
            message = e.response["Error"].get("Message", "").lower()
            if e.response["Error"]["Code"] == "ValidationException" and "role" in message and attempt < 6:
                print(f"Knowledge base role not usable yet, retrying ({attempt}/5)...")
                time.sleep(5)
                continue
            raise
    state["kb_id"] = kb_id
    _wait_for(lambda: agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"],
              "ACTIVE", "Knowledge base")
    print(f"Created knowledge base: {kb_id}")

    # 4. Point it at the documents and ingest: Bedrock chunks each file, embeds the chunks, and writes the vectors.
    ds_id = agent.create_data_source(
        knowledgeBaseId=kb_id, name="sample-docs",
        dataSourceConfiguration={"type": "S3", "s3Configuration": {"bucketArn": f"arn:aws:s3:::{bucket}"}},
    )["dataSource"]["dataSourceId"]
    state["data_source_id"] = ds_id
    job_id = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"]["ingestionJobId"]
    print("Ingesting documents...")
    get_job = lambda: agent.get_ingestion_job(  # noqa: E731
        knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id)["ingestionJob"]
    _wait_for(lambda: get_job()["status"], "COMPLETE", "Ingestion job", timeout=600)
    stats = get_job().get("statistics", {})
    print(f"Ingested {stats.get('numberOfNewDocumentsIndexed', 0)} documents "
          f"({stats.get('numberOfDocumentsFailed', 0)} failed)")
    return kb_id


def delete_sample_knowledge_base(session, state):
    """Delete what create_sample_knowledge_base recorded in `state`, in dependency order. Reports what it could not delete."""
    agent = session.client("bedrock-agent")
    s3vectors = session.client("s3vectors")
    s3 = session.client("s3")

    def gone(get):
        try:
            get()
            return False
        except ClientError as e:
            return e.response["Error"]["Code"] == "ResourceNotFoundException"

    def wait_gone(get, label, timeout=180):
        deadline = time.time() + timeout
        while not gone(get):
            if time.time() > deadline:
                raise TimeoutError(f"{label} still exists after {timeout}s")
            time.sleep(3)

    steps = []
    kb_id, ds_id = state.get("kb_id"), state.get("data_source_id")
    if kb_id and ds_id:
        steps.append((f"data source {ds_id}", lambda: (
            agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id),
            wait_gone(lambda: agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id), "Data source"))))
    if kb_id:
        steps.append((f"knowledge base {kb_id}", lambda: (
            agent.delete_knowledge_base(knowledgeBaseId=kb_id),
            wait_gone(lambda: agent.get_knowledge_base(knowledgeBaseId=kb_id), "Knowledge base"))))
    if state.get("index_created"):
        steps.append((f"vector index {state['index']}", lambda: s3vectors.delete_index(
            vectorBucketName=state["vector_bucket"], indexName=state["index"])))
    if state.get("vector_bucket_created"):
        steps.append((f"vector bucket {state['vector_bucket']}", lambda: s3vectors.delete_vector_bucket(
            vectorBucketName=state["vector_bucket"])))
    if state.get("docs_bucket_created"):
        def delete_bucket():
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=state["docs_bucket"]):
                for obj in page.get("Contents", []):
                    s3.delete_object(Bucket=state["docs_bucket"], Key=obj["Key"])
            s3.delete_bucket(Bucket=state["docs_bucket"])
        steps.append((f"S3 bucket {state['docs_bucket']}", delete_bucket))

    failed = []
    for label, delete in steps:
        try:
            delete()
            print(f"Deleted {label}")
        except Exception as e:  # keep going so one failure does not strand the rest
            failed.append(label)
            print(f"Could not delete {label}: {e}")
    if failed:
        print("Delete these manually:", ", ".join(failed))
    return not failed


def delete_kb_role(iam, name=KB_ROLE_NAME):
    """Delete the sample knowledge base service role (inline policy first)."""
    try:
        iam.delete_role_policy(RoleName=name, PolicyName="BedrockFlowsKnowledgeBasePolicy")
        iam.delete_role(RoleName=name)
        print(f"Deleted role: {name}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Role {name} does not exist")


def find_knowledge_base_id(agent, name):
    """Look up a Bedrock knowledge base ID by exact name.

    Returns None (with a warning) if it is not found, not ACTIVE, or a managed
    knowledge base. The Flows knowledge base node does not support managed ones.
    """
    for page in agent.get_paginator("list_knowledge_bases").paginate():
        for kb in page["knowledgeBaseSummaries"]:
            if kb["name"] != name:
                continue
            kb_id = kb["knowledgeBaseId"]
            detail = agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
            if detail.get("knowledgeBaseConfiguration", {}).get("type") == "MANAGED":
                print(f"Knowledge base '{name}' ({kb_id}) is managed; the Flows knowledge base node does not support it.")
                return None
            if kb.get("status") != "ACTIVE":
                print(f"Knowledge base '{name}' ({kb_id}) is {kb.get('status')}, not ACTIVE.")
                return None
            return kb_id
    print(f"No knowledge base named '{name}' found.")
    return None


# ------------------------------------------------------------- lifecycle ---
def create_flow(agent, name, description, definition, role_arn, tags=None):
    """create_flow, retrying while a freshly created IAM role propagates. Returns the flow ID."""
    args = dict(name=name, description=description, definition=definition, executionRoleArn=role_arn)
    if tags:
        args["tags"] = tags
    for attempt in range(1, 7):
        try:
            return agent.create_flow(**args)["id"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            message = e.response["Error"].get("Message", "").lower()
            if code == "ValidationException" and "role" in message and attempt < 6:
                print(f"IAM role not usable yet, retrying ({attempt}/5)...")
                time.sleep(5)
                continue
            raise


def deploy_flow(agent, flow_id, alias_name="live", timeout=300):
    """Prepare the flow, then create a version and an alias. Returns (version, alias_id)."""
    agent.prepare_flow(flowIdentifier=flow_id)
    deadline = time.time() + timeout
    while True:  # prepare_flow is asynchronous
        status = agent.get_flow(flowIdentifier=flow_id)["status"]
        if status == "Prepared":
            break
        if status == "Failed":
            raise RuntimeError(f"Flow {flow_id} failed to prepare")
        if time.time() > deadline:
            raise TimeoutError(f"Flow {flow_id} not prepared after {timeout}s (status: {status})")
        time.sleep(3)
    version = agent.create_flow_version(flowIdentifier=flow_id)["version"]
    alias_id = agent.create_flow_alias(
        flowIdentifier=flow_id,
        name=alias_name,
        routingConfiguration=[{"flowVersion": version}],
    )["id"]
    return version, alias_id


def invoke_flow(runtime, flow_id, alias_id, document, node_name="FlowInputNode", with_node_names=False):
    """Invoke a flow once and return the list of output documents.

    `document` must match the input node's type: a string, a list (Array), or a dict (Object).
    With `with_node_names=True`, return (output node name, document) pairs instead, which
    shows which branch of a flow produced each answer.
    """
    response = runtime.invoke_flow(
        flowIdentifier=flow_id,
        flowAliasIdentifier=alias_id,
        inputs=[{
            "content": {"document": document},
            "nodeName": node_name,
            "nodeOutputName": "document",
        }],
    )
    outputs, status = [], None
    for event in response["responseStream"]:
        if "flowOutputEvent" in event:
            output = event["flowOutputEvent"]
            document_out = output["content"]["document"]
            outputs.append((output["nodeName"], document_out) if with_node_names else document_out)
        elif "flowCompletionEvent" in event:
            status = event["flowCompletionEvent"]["completionReason"]
    if status != "SUCCESS":
        raise RuntimeError(f"Flow ended with status {status}")
    return outputs


def cleanup_flow(agent, flow_id=None, alias_id=None, version=None):
    """Delete the alias, then the version, then the flow. Reports anything it could not delete."""
    steps = [
        ("alias", alias_id, lambda: agent.delete_flow_alias(flowIdentifier=flow_id, aliasIdentifier=alias_id)),
        ("version", version, lambda: agent.delete_flow_version(flowIdentifier=flow_id, flowVersion=version)),
        ("flow", flow_id, lambda: agent.delete_flow(flowIdentifier=flow_id)),
    ]
    failed = []
    for label, value, delete in steps:
        if not value:
            continue
        try:
            delete()
            print(f"Deleted {label}: {value}")
        except Exception as e:  # keep going so one failure does not strand the rest
            failed.append(f"{label} {value}")
            print(f"Could not delete {label} {value}: {e}")
    if failed:
        print("Delete these manually:", ", ".join(failed))
    return not failed
