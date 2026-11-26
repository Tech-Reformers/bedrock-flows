"""Condition node: route a request down different paths.

A first prompt classifies the question as AWS or OTHER. The Condition node then
sends AWS questions to a Lambda node that searches AWS documentation through the
public AWS Knowledge MCP Server, and an "AWS expert" prompt answers from those
excerpts with source URLs. Everything else goes to a general prompt.

The script deploys the Lambda function (aws_docs_lambda.py) for you and deletes
it at the end, along with the flow.

Usage:
    python conditions.py                                      # one AWS and one general question
    python conditions.py --question "How do I invoke a Bedrock flow from my app?"
    python conditions.py --keep                               # leave the flow and Lambda in place
    python conditions.py --delete-role                        # also delete the sample IAM roles
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402

HERE = Path(__file__).resolve().parent
LAMBDA_NAME = "BedrockFlowsSampleAwsDocsSearch"
DEFAULT_QUESTIONS = [
    "How do I invoke a Bedrock flow from my application?",
    "Give me a one-sentence tip for staying focused.",
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--question", action="append", help="a question to send (repeat for several)")
parser.add_argument("--model-id", help="model or inference profile ID (default: Amazon Nova 2 Lite)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow and Lambda function when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM roles at the end")
args = parser.parse_args()

session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")
runtime = session.client("bedrock-agent-runtime")
lambda_client = session.client("lambda")
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_model_id(region, account)
print(f"Account {account} | region {region} | model {model_id}")

role_arn = flow_utils.ensure_role(iam, region, account)

lambda_arn = flow_id = version = alias_id = None
try:
    # The Lambda node's function: searches AWS docs through the AWS Knowledge MCP Server.
    lambda_arn = flow_utils.deploy_lambda(lambda_client, iam, LAMBDA_NAME, HERE / "aws_docs_lambda.py")

    template = flow_utils.load_template(HERE / "conditions_flow.json", {
        "$$PROMPT_CLASSIFY_INPUT_MODEL_ID": model_id,
        "$$PROMPT_MODEL_ID": model_id,
        "$$DOCS_LAMBDA_ARN": lambda_arn,
    })
    flow_id = flow_utils.create_flow(
        agent, template["name"], template["description"], template["definition"], role_arn, template.get("tags"))
    version, alias_id = flow_utils.deploy_flow(agent, flow_id)

    for question in args.question or DEFAULT_QUESTIONS:
        outputs = flow_utils.invoke_flow(runtime, flow_id, alias_id, question, with_node_names=True)
        for node_name, answer in outputs:
            print(f"\nQ: {question}\nAnswered by: {node_name}\nA: {answer}")
finally:
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id}) and Lambda function {LAMBDA_NAME}.")
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
        if lambda_arn:
            flow_utils.delete_lambda(lambda_client, LAMBDA_NAME)
    if args.delete_role:
        flow_utils.delete_role(iam)
        flow_utils.delete_lambda_role(iam)
