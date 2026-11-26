"""Knowledge Base node: answer from your documents.

The script builds a knowledge base for you on Amazon S3 Vectors from the sample
documents in sample_docs/ (policies for Octank Outfitters, a made-up outdoor gear
store), then runs a flow whose Knowledge Base node retrieves matching chunks and
has a model write a grounded answer. At the end it deletes the flow and every
knowledge base resource it created.

What gets created:
    S3 bucket          the documents (private)
    S3 vector bucket   with one vector index for the embeddings
    IAM role           BedrockFlowsSampleKnowledgeBaseRole, used by the knowledge base
    Knowledge base     with an S3 data source, filled by one ingestion job

Setup takes a few minutes, mostly the ingestion job. You pay for the embedding
calls, the vector storage, and the model calls, all small for three documents.

The Flows knowledge base node does not support *managed* knowledge bases, which
is why this builds one on S3 Vectors. To use a knowledge base you already have
(not managed), pass --knowledge-base-id or --knowledge-base-name instead.

Usage:
    python knowledge_base.py                                   # build the sample knowledge base and ask
    python knowledge_base.py --question "Can I return worn hiking boots?"
    python knowledge_base.py --knowledge-base-name my-kb --question "What does the report cover?"
    python knowledge_base.py --keep                            # leave the flow and knowledge base in place
    python knowledge_base.py --delete-role                     # also delete the sample IAM roles
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = [
    "Can I return hiking boots I've already worn on a trail?",
    "How much does it cost to ship a kayak?",
    "What is the warranty on an Octank jacket?",
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
existing = parser.add_mutually_exclusive_group()
existing.add_argument("--knowledge-base-id", help="use your own (non-managed) knowledge base instead of the sample")
existing.add_argument("--knowledge-base-name", help="look up your own (non-managed) knowledge base by name")
parser.add_argument("--question", action="append", help="a question to ask (repeat for several)")
parser.add_argument("--model-id", help="model or inference profile ID (default: Amazon Nova 2 Lite)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow and knowledge base when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM roles at the end")
args = parser.parse_args()

session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")
runtime = session.client("bedrock-agent-runtime")
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_model_id(region, account)
print(f"Account {account} | region {region} | model {model_id}")

role_arn = flow_utils.ensure_role(iam, region, account)

kb_state = {}  # what create_sample_knowledge_base made, for cleanup
flow_id = version = alias_id = None
try:
    if args.knowledge_base_id or args.knowledge_base_name:
        kb_id = args.knowledge_base_id or flow_utils.find_knowledge_base_id(agent, args.knowledge_base_name)
        if not kb_id:
            sys.exit("No usable knowledge base. See the message above.")
        questions = args.question or ["What is this knowledge base about?"]
    else:
        kb_id = flow_utils.create_sample_knowledge_base(session, region, account, HERE / "sample_docs", kb_state)
        questions = args.question or DEFAULT_QUESTIONS

    template = flow_utils.load_template(HERE / "rag_kb_flow.json", {
        "$$KNOWLEDGEBASE_ID": kb_id,
        "$$KNOWLEDGEBASE_MODEL_ID": model_id,
    })
    flow_id = flow_utils.create_flow(
        agent, template["name"], template["description"], template["definition"], role_arn, template.get("tags"))
    version, alias_id = flow_utils.deploy_flow(agent, flow_id)

    for question in questions:
        outputs = flow_utils.invoke_flow(runtime, flow_id, alias_id, question)
        print(f"\nQ: {question}\nA: {outputs[0]}")
finally:
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id})"
              + (f" and knowledge base {kb_state.get('kb_id')}." if kb_state else "."))
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
        if kb_state:
            flow_utils.delete_sample_knowledge_base(session, kb_state)
    if args.delete_role:
        flow_utils.delete_role(iam)
        flow_utils.delete_kb_role(iam)
