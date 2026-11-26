# Amazon Bedrock Flows: Teaching and Learning Samples

Short, runnable demos for learning **Amazon Bedrock Flows**: build a workflow by
wiring nodes (prompts, conditions, iterators, guardrails, agents, knowledge bases)
together with connections, then deploy it and call it from code.

Each module has the same pieces:

- **`.py` script**: run from the command line, shows the code as a real Python file
- **`.ipynb` notebook**: run cell by cell, with explanations and *Teaching/Learning Tips*
- **a flow template** (`*_flow.json`), and a diagram of the flow where the module has one

Nothing is hard-coded to an account or region. Both come from your current AWS login and configuration.

## Where flows fit in a real application

A flow is a versioned AI pipeline that your application calls like an API. You build the steps once
(prompts, knowledge base lookups, conditions, guardrails, agents), publish a version, and point an alias
at it. Callers always use the alias, so you can ship a new version or roll back without changing the app.

Typical uses:

- **Question routing:** classify a question, then send AWS questions to an expert grounded in AWS documentation and everything else to a general answer (see `03_conditions`). The same shape handles ticket triage: route billing, outages, and general requests to different steps.
- **Internal Q&A:** guardrail, knowledge base retrieval, then an answer with citations, called from a chatbot (see `04_guardrails` and `06_knowledge_base`).
- **Batch processing:** summarize or extract fields from a list of documents, one item at a time (see `02_iterator_collector`).

### How an app calls a flow

Every caller ends up using the [`InvokeFlow`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeFlow.html)
API on a flow alias. It streams results back, and the AWS CLI does not support it, so you call it from an AWS SDK.
The caller needs IAM permission to invoke that flow alias.

| Caller | How it calls the flow |
| --- | --- |
| Backend code you control (a server, Lambda, or job) | Calls `invoke_flow` on the `bedrock-agent-runtime` client directly, as the demos do. |
| Browser, mobile app, webhook, or partner system | API Gateway (with Cognito, API keys, or a Lambda authorizer) in front of a small Lambda that calls `invoke_flow`. API Gateway does not call a flow without that Lambda. |
| Events such as an S3 upload, SQS message, or schedule | A Lambda triggered by the event calls `invoke_flow`. |

A flow is a poor fit for a single model call (use the Converse API), for logic with complex state or loops
between custom code steps (consider AWS Step Functions), or for very latency-sensitive paths, since each node adds time.

## Requirements

- Python 3.10 or newer
- An AWS account with access to Amazon Bedrock. Use a sandbox account, not production.
- AWS credentials configured on your machine, and a default region (see [AWS credentials](#aws-credentials)). Tested in `us-west-2`.
- Permission to create IAM roles and the resources the modules use: flows, a guardrail (04), Lambda functions (03, 05), agents (05), and a knowledge base with S3 and S3 Vectors buckets (06). Every script deletes what it creates.
- **Model access.** The modules use Amazon Nova 2 Lite, Claude Haiku 4.5 (module 05), and Amazon Titan Text Embeddings V2 (module 06). Serverless models are enabled by default, but Anthropic requires a one-time use case form before first use: open any Claude model in the Bedrock console playground and submit it.
- **Module 05 only:** Amazon Bedrock Agents Classic, which is closed to new customers (see [module 05](#05_agents)).

## Setup

Get the code:

```bash
git clone https://github.com/Tech-Reformers/bedrock-flows.git
cd bedrock-flows
```

Then run the following once from the project folder. It creates an isolated Python environment (a *venv*) so the
demo dependencies do not affect the rest of your system.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run this once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## AWS credentials

Any normal way of signing in works: `aws configure`, `aws sso login`, `aws login`, or environment
variables. To pick a named profile, set `AWS_PROFILE`. The region comes from `AWS_REGION` or your
profile's configured region. Set one if you have not:

```bash
aws configure set region <your-region>
```

The demos use **Amazon Nova 2 Lite** through a cross-region inference profile. `flow_utils` looks up
the profile your region offers (for example `us.`, `eu.`, or `jp.`, or `global.` where there is no
geographic one) and passes its ARN, as the Flows docs recommend. The bare profile ID also runs, but the
console's prompt node editor flags it as an invalid model identifier. Pass `--model-id` to use a different model.

## Running a demo

Run a script from inside its folder:

```bash
cd 01_flow_lifecycle
python flow_lifecycle.py
```

Every script creates what it needs, runs, and **deletes everything it created**, even if a step fails. Useful flags:

| Flag | What it does |
| --- | --- |
| `--keep` | Leave the flow (and any agents, Lambda function, or knowledge base) in place so you can open it in the Bedrock console |
| `--delete-role` | Also delete the sample IAM roles at the end |
| `--model-id <id>` | Use a different model or inference profile |

To run a notebook, open the `.ipynb` file in VS Code or Kiro, select the `venv` as the kernel, and run
the cells with Shift+Enter. Run the **Cleanup** cell at the end. Unlike the scripts, a notebook can't
clean up for you if you stop partway, so run Cleanup even after an error.

## Modules

Modules 01 to 06 each teach one idea, and each builds on the one before it. The [Tools](#tools) section covers the supporting files.

### `01_flow_lifecycle/`

| File | What it shows |
| --- | --- |
| `flow_lifecycle.py` | The smallest useful flow (input, prompt, output) and every API call in order: create, prepare, version, alias, invoke, delete. |
| `flow_lifecycle.ipynb` | The same lifecycle cell by cell, with notes on the console builder and on drafts versus aliases. |

*Teaching/Learning Tip:* a flow is orchestration you *configure* rather than code. The console builder and the API are two views of the same definition. You invoke a stable **alias**, so you can roll versions forward or back without changing the caller, the same idea as a Lambda alias.

### `02_iterator_collector/`

| File | What it shows |
| --- | --- |
| `iterator_collector.py` | An Iterator node fans an array out, a Prompt node runs once per item, a Collector node fans the results back in. |
| `iterator_collector.ipynb` | Notebook walkthrough with the same flow. |
| `iterator_collector_flow.json` | The flow definition. |

*Teaching/Learning Tip:* this is fan-out/fan-in, the pattern you would otherwise hand-write with Step Functions or queue code. The Input node is typed Array, so the input must be a JSON array, not plain text.

### `03_conditions/`

| File | What it shows |
| --- | --- |
| `conditions.py` | A classifier prompt labels the question `AWS` or `OTHER`, and a Condition node routes it. AWS questions go to a Lambda node that searches AWS documentation through the public [AWS Knowledge MCP Server](https://aws.amazon.com/about-aws/whats-new/2025/10/aws-knowledge-mcp-server-generally-available/), then to an AWS expert prompt that answers from those excerpts with source URLs. Everything else goes to a general prompt. Deploys and deletes the Lambda function for you. |
| `conditions.ipynb` | Notebook walkthrough. |
| `conditions_flow.json` | The flow definition. |
| `aws_docs_lambda.py` | The Lambda function. It calls the MCP server's `aws___search_documentation` tool with the standard library only. Module 05 reuses it as an agent tool, so it handles both the flow node event and the agent action group event. |

*Teaching/Learning Tip:* a Condition node takes the first matching expression, with a `default` branch as the fallback, and the comparison is exact. Keep the classifier prompt strict ("reply with exactly one word") and cap `maxTokens`, or the model adds an explanation and the condition never matches.

*Teaching/Learning Tip:* a prompt node cannot call tools or MCP servers. To bring outside data into a flow, call it from a Lambda node and wire the result into the prompt as a variable.

### `04_guardrails/`

| File | What it shows |
| --- | --- |
| `guardrails.py` | Creates a guardrail that denies investment advice, attaches it to a Prompt node, and sends one allowed and one blocked question. Deletes the guardrail at the end. |
| `guardrails.ipynb` | Notebook walkthrough. |
| `prompt_guardrail_flow.json` | The flow definition. |

*Teaching/Learning Tip:* the flow only references the guardrail by ID and version. A blocked input never reaches the model, and the caller gets the guardrail's configured message.

### `05_agents/`

Creates its own two agents and their tool, so there is nothing to set up first.

| File | What it shows |
| --- | --- |
| `agents.py` | Creates a Researcher agent with one tool, `search_aws_docs` (the module 03 Lambda function, used here as an agent action group), and an Editor agent with no tools. It calls the Researcher directly with tracing on so you can watch it decide when and what to search, then chains both agents in a flow. Deletes the flow, agents, and Lambda function at the end. |
| `agents.ipynb` | Notebook walkthrough. |
| `multi_agent_flow.json` | The two-agent flow definition. |
| `multi_turn_agent_flow.json` | A flow that holds a back-and-forth conversation with one agent. It needs interactive input, so run it with the flow manager (`tools/flow_manager`). |

*Teaching/Learning Tip:* compare with module 03. There, a Condition node decides the route and every AWS question gets one search. Here, the agent decides: it skips the tool for a non-AWS question, and for an AWS question it may search, judge the results, and search again with a narrower query.

*Teaching/Learning Tip:* the agents use Claude Haiku 4.5, not Nova 2 Lite. With Agents Classic's default orchestration prompt, Nova 2 Lite sometimes ends a turn with neither a tool call nor an answer, and the agent replies "Sorry, I am unable to assist you with this request." Newer Claude models (for example Haiku 5.5) reject the thinking settings Agents Classic sends.

*Teaching/Learning Tip:* a flow calls an agent *alias*, not the agent, so you can move the alias to a new agent version without touching the flow. The console's test alias (`TSTALIASID`) does not work in a flow.

> **Availability:** Amazon Bedrock Agents (now Agents Classic) is in [maintenance mode](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html) and closed to new customers. Accounts that already use it can still create agents. If `create_agent` fails for your account, skip this module. Amazon Bedrock AgentCore is the recommended platform for new agent projects.

### `06_knowledge_base/`

Builds its own knowledge base on Amazon S3 Vectors, so there is nothing to set up first.

| File | What it shows |
| --- | --- |
| `knowledge_base.py` | Creates a knowledge base from the documents in `sample_docs/` (an S3 bucket, an S3 vector bucket and index, a service role, a data source, and an ingestion job), answers questions from it through a Knowledge Base node, then deletes the flow and every knowledge base resource. Pass `--knowledge-base-id` or `--knowledge-base-name` to use your own knowledge base instead. |
| `knowledge_base.ipynb` | Notebook walkthrough, one step per cell. |
| `rag_kb_flow.json` | The flow definition. |
| `sample_docs/` | Three short policy documents for Octank Outfitters, a made-up outdoor gear store, so answers clearly come from the documents. |

*Teaching/Learning Tip:* this is the grounding idea behind RAG: give the model your documents as context so it answers from them, not from its training memory.

*Teaching/Learning Tip:* the Flows knowledge base node does not support *managed* knowledge bases (the invoke fails with "This operation is not supported for managed knowledge bases"). It needs one backed by a vector store you own, which is why this module builds one on S3 Vectors.

Setup takes a few minutes, mostly the ingestion job. The costs (embedding calls, vector storage, model calls) are small for three documents.

## Tools

Support files for working with flows outside the numbered modules.

### `tools/flow_manager/`

| File | What it shows |
| --- | --- |
| `bedrock_flow_manager.py` | An interactive CLI that finds every `*_flow.json` template in the repo, fills in its `$$VARIABLES`, creates and tests the flow, and cleans up. Supports multi-turn flows. |

```bash
python tools/flow_manager/bedrock_flow_manager.py --cleanup --test-input "Hello"
python tools/flow_manager/bedrock_flow_manager.py --help
```

Useful options: `--region`, `--profile`, `--flow-name`, `--existing-role`, `--templates-dir`, and
`--knowledge-base-name` (look up a knowledge base by name for templates that need one). For iterator flows, pass several values to `--test-input`.

*Teaching/Learning Tip:* once you understand the lifecycle (module 01), a template plus a small tool is all you need to try any flow definition quickly.

### `tools/cloudformation/`

| File | What it shows |
| --- | --- |
| `bedrock_flows.json` | The iterator/collector flow as an `AWS::Bedrock::Flow`, with a `FlowVersion` and `FlowAlias`, deployed with CloudFormation. |

```bash
aws cloudformation deploy --stack-name flows-cfn-demo \
  --template-file tools/cloudformation/bedrock_flows.json \
  --parameter-overrides ModelId=arn:aws:bedrock:<region>:<account-id>:inference-profile/us.amazon.nova-2-lite-v1:0
```

The flow assumes an existing role. Run any module script once to create `BedrockFlowsSampleRole`, or pass your own with `RoleName=`.
Delete the stack when you are done: `aws cloudformation delete-stack --stack-name flows-cfn-demo`.

## The shared helper: `flow_utils.py`

Module 01 shows every Flows API call by hand. From module 02 on, the scripts and notebooks import
`flow_utils.py` for the repeated plumbing: creating the IAM role, the create-prepare-version-alias
lifecycle (including waiting for `Prepared`), invoking a flow, and cleanup. Read it once. It is short.

### The IAM role

`BedrockFlowsSampleRole` is the *execution role* a flow runs under:

- **Trust:** only Amazon Bedrock flows in your account and region can assume it (`aws:SourceAccount`, `aws:SourceArn`).
- **Permissions:** runtime actions only, scoped to your account and region where the ARN allows: invoke models, get prompts, query knowledge bases, invoke agents, apply guardrails, and invoke Lambda functions whose names start with `BedrockFlowsSample`.

Module 03 also creates `BedrockFlowsSampleLambdaRole`, the execution role for its Lambda function. It has CloudWatch Logs permissions only (`AWSLambdaBasicExecutionRole`).
Module 05 creates `BedrockFlowsSampleAgentRole`, the service role for its agents. Its only permissions are invoking models and reading their inference profiles. The Researcher agent calls its Lambda tool through a resource-based policy on the function, scoped to that agent.
Module 06 creates `BedrockFlowsSampleKnowledgeBaseRole`, the service role for its knowledge base. It can read the sample documents bucket, call the embedding model, and read and write the one vector index.

The role is created if missing and its policies are refreshed on every run. Delete it with `--delete-role` (modules 03, 05, and 06 also delete their Lambda, agent, and knowledge base roles).

## Cost and cleanup

Scripts delete the flows they create, module 03 deletes its Lambda function, module 05 deletes its agents and Lambda function, and module 06 deletes its knowledge base, vector store, and documents bucket. The sample IAM roles are kept for reuse (they are free).

Flows, agents, and Lambda functions have no charge when idle. You pay for model calls, and in module 06
for embeddings and a small amount of S3 and S3 Vectors storage while it runs.

If a notebook stops before its Cleanup cell, or you used `--keep`, delete leftovers. Everything the
modules create has a name starting with `BedrockFlowsSample`, `bedrock-flows-sample`, or `sample-`:

```bash
aws bedrock-agent list-flows                      # flows
aws bedrock-agent list-agents                     # module 05
aws bedrock-agent list-knowledge-bases            # module 06
aws lambda list-functions --query "Functions[?starts_with(FunctionName,'BedrockFlowsSample')].FunctionName"
aws s3vectors list-vector-buckets                 # module 06
aws s3 ls | grep bedrock-flows-sample             # module 06
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No AWS region found` | Set `AWS_REGION` or `aws configure set region <region>`. |
| `temperature and top_p cannot both be specified` | Newer Claude models reject both. The templates here set `temperature` only. |
| `Could not assume role` right after the role was created | IAM is eventually consistent. The scripts retry for about 30 seconds. |
| `not authorized to perform: bedrock:ApplyGuardrail` on a `guardrail-profile` | The role needs `guardrail-profile/*`. `flow_utils.ensure_role` refreshes it, so re-run a module. |
| `not supported for managed knowledge bases` | The Flows knowledge base node needs a non-managed knowledge base. Module 06 builds one on S3 Vectors. |
| Flow fails with `not prepared` | Flows must reach `Prepared` before you create a version. The helpers wait for it. |
| `AccessDeniedException` calling a Claude model | Anthropic needs a one-time use case form per account. Open a Claude model in the Bedrock console playground and submit it. |
| `ConflictException` ... `already exists` | A resource with the same name is left over from an interrupted run or `--keep`. Delete it (see [Cost and cleanup](#cost-and-cleanup)) and rerun. |
| Module 05 agent replies "Sorry, I am unable to assist you with this request." | The model didn't follow the agent's response format. Check the trace for `LLM prediction format incorrect`. Use the default Claude Haiku 4.5. |
| Module 03 or 05 answers "No AWS documentation was found" | The AWS Knowledge MCP Server is public and rate limited. Wait a minute and retry. |

## Credits and license

Based on [`aws-samples/amazon-bedrock-flows-samples`](https://github.com/aws-samples/amazon-bedrock-flows-samples)
(Amit Lulla, Li Chen, Naresh Nagpal), originally released under the MIT No Attribution (MIT-0) license and reworked into a teaching repo. This repo is licensed under the MIT License. See [LICENSE](LICENSE).
