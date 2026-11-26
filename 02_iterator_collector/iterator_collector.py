"""Iterator and Collector nodes: fan out over an array, fan back in.

The Iterator node splits an array and sends each item downstream one at a time.
The Prompt node runs once per item. The Collector node waits for every
iteration and reassembles the results into one array.

Usage:
    python iterator_collector.py
    python iterator_collector.py --question "What is S3?" --question "What is Lambda?"
    python iterator_collector.py --keep          # leave the flow in place
    python iterator_collector.py --delete-role   # also delete the sample IAM role
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = [
    "What is Amazon Bedrock in one sentence?",
    "Write a haiku about cloud computing.",
    "Name three AWS regions.",
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--question", action="append", help="a question to send (repeat for several)")
parser.add_argument("--model-id", help="model or inference profile ID (default: Amazon Nova 2 Lite)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM role at the end")
args = parser.parse_args()

session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")
runtime = session.client("bedrock-agent-runtime")
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_model_id(region, account)
print(f"Account {account} | region {region} | model {model_id}")

role_arn = flow_utils.ensure_role(iam, region, account)
template = flow_utils.load_template(HERE / "iterator_collector_flow.json", {"$$PROMPT_MODEL_ID": model_id})

flow_id = version = alias_id = None
try:
    flow_id = flow_utils.create_flow(
        agent, template["name"], template["description"], template["definition"], role_arn, template.get("tags"))
    version, alias_id = flow_utils.deploy_flow(agent, flow_id)

    # The Input node is typed Array, so the input must be a JSON array, not plain text.
    questions = args.question or DEFAULT_QUESTIONS
    outputs = flow_utils.invoke_flow(runtime, flow_id, alias_id, questions)

    for question, answer in zip(questions, outputs[0]):
        print(f"\nQ: {question}\nA: {answer}")
finally:
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id}).")
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
    if args.delete_role:
        flow_utils.delete_role(iam)
