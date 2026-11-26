"""The Amazon Bedrock Flows lifecycle, one API call at a time.

Builds the smallest useful flow (input -> prompt -> output), then walks through
create -> prepare -> version -> alias -> invoke -> delete, calling each boto3
API directly so you can see exactly what happens.

Usage:
    python flow_lifecycle.py                    # run the whole lifecycle, then clean up
    python flow_lifecycle.py --keep             # leave the flow in place (so you can open it in the console)
    python flow_lifecycle.py --model-id <id>    # use a different model or inference profile
    python flow_lifecycle.py --delete-role      # also delete the sample IAM role at the end
"""
import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402  (shared IAM role helper)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model-id", help="model or inference profile ID (default: Amazon Nova 2 Lite)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM role at the end")
args = parser.parse_args()

# 1. Clients. The region and credentials come from your AWS configuration.
session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")                  # control plane: create/manage flows
runtime = session.client("bedrock-agent-runtime")        # data plane: invoke flows
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_model_id(region, account)
print(f"Account {account} | region {region} | model {model_id}")

# 2. The execution role Bedrock assumes while the flow runs.
role_arn = flow_utils.ensure_role(iam, region, account)

# 3. The flow definition: three nodes joined by two connections.
definition = {
    "nodes": [
        {"name": "FlowInputNode", "type": "Input",
         "outputs": [{"name": "document", "type": "String"}]},
        {"name": "Explain", "type": "Prompt",
         "inputs": [{"name": "topic", "type": "String", "expression": "$.data"}],
         "outputs": [{"name": "modelCompletion", "type": "String"}],
         "configuration": {"prompt": {"sourceConfiguration": {"inline": {
             "modelId": model_id,
             "templateType": "TEXT",
             "inferenceConfiguration": {"text": {"temperature": 0.7}},
             "templateConfiguration": {"text": {"text": "Explain {{topic}} in two sentences."}},
         }}}}},
        {"name": "FlowOutputNode", "type": "Output",
         "inputs": [{"name": "document", "type": "String", "expression": "$.data"}]},
    ],
    "connections": [
        {"name": "InputToExplain", "source": "FlowInputNode", "target": "Explain", "type": "Data",
         "configuration": {"data": {"sourceOutput": "document", "targetInput": "topic"}}},
        {"name": "ExplainToOutput", "source": "Explain", "target": "FlowOutputNode", "type": "Data",
         "configuration": {"data": {"sourceOutput": "modelCompletion", "targetInput": "document"}}},
    ],
}

flow_id = version = alias_id = None
try:
    # 4. create_flow registers the definition (a DRAFT you can still edit).
    for attempt in range(6):  # a brand-new IAM role can take a few seconds to become usable
        try:
            flow_id = agent.create_flow(
                name="LifecycleDemoFlow",
                description="Smallest useful flow: input -> prompt -> output",
                executionRoleArn=role_arn,
                definition=definition,
            )["id"]
            break
        except agent.exceptions.ValidationException as e:
            if "role" not in str(e).lower() or attempt == 5:
                raise
            time.sleep(5)
    print("flow_id:", flow_id)

    # 5. prepare_flow compiles and validates the draft. It is asynchronous, so poll until done.
    agent.prepare_flow(flowIdentifier=flow_id)
    while True:
        status = agent.get_flow(flowIdentifier=flow_id)["status"]
        print("status:", status)
        if status == "Prepared":
            break
        if status == "Failed":
            raise RuntimeError("Flow failed to prepare")
        time.sleep(3)

    # 6. create_flow_version snapshots the prepared draft into an immutable version.
    version = agent.create_flow_version(flowIdentifier=flow_id)["version"]
    print("version:", version)

    # 7. create_flow_alias points a stable name at that version. Callers invoke the alias.
    alias_id = agent.create_flow_alias(
        flowIdentifier=flow_id,
        name="live",
        routingConfiguration=[{"flowVersion": version}],
    )["id"]
    print("alias_id:", alias_id)

    # 8. invoke_flow streams events back. The input must match the Input node's type (a string here).
    response = runtime.invoke_flow(
        flowIdentifier=flow_id,
        flowAliasIdentifier=alias_id,
        inputs=[{
            "content": {"document": "Amazon Bedrock Flows"},
            "nodeName": "FlowInputNode",
            "nodeOutputName": "document",
        }],
    )
    for event in response["responseStream"]:
        if "flowOutputEvent" in event:
            print("\nOUTPUT:", event["flowOutputEvent"]["content"]["document"])
        elif "flowCompletionEvent" in event:
            print("completion:", event["flowCompletionEvent"]["completionReason"])

finally:
    # 9. Delete in order: alias, version, flow. (An alias or version blocks deleting what it points to.)
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id}). Delete it later with the console or delete_flow.")
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
    if args.delete_role:
        flow_utils.delete_role(iam)
