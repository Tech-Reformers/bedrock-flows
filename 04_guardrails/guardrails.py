"""Guardrails on a Prompt node: block topics before they reach the model.

This script creates a small guardrail that denies investment advice, builds a
flow whose Prompt node uses it, and sends two questions: one allowed, one
blocked. The guardrail is deleted at the end.

Usage:
    python guardrails.py
    python guardrails.py --keep          # leave the flow and guardrail in place
    python guardrails.py --delete-role   # also delete the sample IAM role
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = [
    "Tell me a short fun fact about the ocean.",       # allowed
    "Which stocks should I buy to get rich quickly?",  # blocked by the guardrail
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model-id", help="model or inference profile ID (default: Amazon Nova 2 Lite)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow and guardrail when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM role at the end")
args = parser.parse_args()

session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")
runtime = session.client("bedrock-agent-runtime")
bedrock = session.client("bedrock")  # control plane for guardrails
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_model_id(region, account)
print(f"Account {account} | region {region} | model {model_id}")

role_arn = flow_utils.ensure_role(iam, region, account)

guardrail_id = flow_id = version = alias_id = None
try:
    # A guardrail with one denied topic. Messages are what the caller sees when it blocks.
    guardrail_id = bedrock.create_guardrail(
        name="flows-sample-guardrail",
        description="Sample guardrail for the Bedrock Flows teaching repo",
        topicPolicyConfig={"topicsConfig": [{
            "name": "investment-advice",
            "definition": "Recommendations about buying or selling stocks, crypto, or other investments.",
            "examples": ["Which stock should I buy?", "Is Bitcoin a good investment?"],
            "type": "DENY",
        }]},
        blockedInputMessaging="Sorry, I can't help with investment advice.",
        blockedOutputsMessaging="Sorry, I can't help with investment advice.",
    )["guardrailId"]
    print("guardrail_id:", guardrail_id)

    template = flow_utils.load_template(HERE / "prompt_guardrail_flow.json", {
        "$$PROMPT_MODEL_ID": model_id,
        "$$GUARDRAIL_ID": guardrail_id,
    })
    flow_id = flow_utils.create_flow(
        agent, template["name"], template["description"], template["definition"], role_arn, template.get("tags"))
    version, alias_id = flow_utils.deploy_flow(agent, flow_id)

    for question in QUESTIONS:
        outputs = flow_utils.invoke_flow(runtime, flow_id, alias_id, question)
        print(f"\nQ: {question}\nA: {outputs[0]}")
finally:
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id}) and guardrail {guardrail_id}.")
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
        if guardrail_id:
            bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)
            print(f"Deleted guardrail: {guardrail_id}")
    if args.delete_role:
        flow_utils.delete_role(iam)
