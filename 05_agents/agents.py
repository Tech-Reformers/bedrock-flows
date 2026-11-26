"""Agent nodes: call Bedrock Agents from a flow.

multi_agent_flow.json chains two agents: the first agent's answer becomes the
second agent's input. This script creates both agents for you:

    Researcher: has one tool, search_aws_docs, which searches AWS documentation
                through the public AWS Knowledge MCP Server (the same Lambda
                function as module 03). The agent decides whether to search,
                what to search for, and whether to search again.
    Editor:     rewrites the Researcher's answer as three bullet points plus sources.

It first calls the Researcher directly with tracing on, so you can watch it
decide (an AWS question uses the tool, a general one does not). Then it builds
the flow, invokes it, and deletes the flow, both agents, and the Lambda function.

Compare with module 03: there the flow decides the route with a Condition node.
Here the agent decides for itself.

Amazon Bedrock Agents (now "Agents Classic") is in maintenance mode and closed
to new customers. Accounts that already use it can still create agents. If
create_agent fails for your account, skip this module (see Amazon Bedrock
AgentCore for the newer agent platform).

The second template, multi_turn_agent_flow.json, holds a back-and-forth
conversation with one agent. It needs interactive input, so run it with the
flow manager in tools/flow_manager.

Usage:
    python agents.py                                   # default questions
    python agents.py --question "How do I give a Bedrock flow access to a Lambda function?"
    python agents.py --keep                            # leave the flow, agents, and Lambda in place
    python agents.py --delete-role                     # also delete the sample IAM roles
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flow_utils  # noqa: E402

HERE = Path(__file__).resolve().parent
LAMBDA_NAME = "BedrockFlowsSampleAgentDocsTool"
LAMBDA_SOURCE = HERE.parent / "03_conditions" / "aws_docs_lambda.py"  # shared with module 03

RESEARCHER = ("BedrockFlowsSampleResearcher",
              "You are an AWS expert researcher. For questions about AWS, use the search_aws_docs tool to find "
              "relevant AWS documentation before you answer. If the results don't answer the question, search "
              "again with a narrower query, but search at most three times, then answer with what you found. For questions that are not about AWS, answer from your own knowledge "
              "without the tool. Answer in one short paragraph of plain text, then list the source URLs you used.")
EDITOR = ("BedrockFlowsSampleEditor",
          "You are an editor with no tools. The user's message is a draft answer. Rewrite it as exactly three "
          "concise bullet points that keep its key facts. If the draft lists source URLs, keep them in a "
          "'Sources:' list after the bullets. Return only the bullets and sources.")
DOCS_TOOL = {
    "actionGroupName": "aws_docs",
    "description": "Search official AWS documentation.",
    "functionSchema": {"functions": [{
        "name": "search_aws_docs",
        "description": "Search AWS documentation, blog posts, and What's New posts. Returns excerpts with source URLs.",
        "parameters": {"query": {
            "type": "string",
            "description": "Search phrase, for example 'Bedrock flow Lambda node permissions'.",
            "required": True,
        }},
    }]},
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--question", help="text sent to the flow (default: an AWS question)")
parser.add_argument("--model-id", help="model or inference profile for the agents (default: Claude Haiku 4.5)")
parser.add_argument("--keep", action="store_true", help="do not delete the flow, agents, and Lambda when finished")
parser.add_argument("--delete-role", action="store_true", help="delete the sample IAM roles at the end")
args = parser.parse_args()

session, region, account = flow_utils.get_session_info()
agent = session.client("bedrock-agent")
runtime = session.client("bedrock-agent-runtime")
lambda_client = session.client("lambda")
iam = session.client("iam")
model_id = args.model_id or flow_utils.default_agent_model_id(region, account)
print(f"Account {account} | region {region} | agent model {model_id}")

role_arn = flow_utils.ensure_role(iam, region, account)  # the flow's role: may invoke agent aliases
agent_role_arn = flow_utils.ensure_agent_role(iam, region, account)  # the agents' role: may invoke the model

created = []  # (agent_id, alias_id) for cleanup
lambda_arn = flow_id = version = alias_id = None
try:
    # The Researcher's tool. The agent calls it through a resource-based policy on the function.
    lambda_arn = flow_utils.deploy_lambda(lambda_client, iam, LAMBDA_NAME, LAMBDA_SOURCE)
    tool = dict(DOCS_TOOL, actionGroupExecutor={"lambda": lambda_arn})
    researcher_id, researcher_alias_id, researcher_arn = flow_utils.create_agent(
        agent, *RESEARCHER, model_id, agent_role_arn, action_groups=[tool],
        before_prepare=lambda agent_id: flow_utils.allow_agent_to_invoke_lambda(
            lambda_client, LAMBDA_NAME, region, account, agent_id))
    created.append((researcher_id, researcher_alias_id))
    editor_id, editor_alias_id, editor_arn = flow_utils.create_agent(agent, *EDITOR, model_id, agent_role_arn)
    created.append((editor_id, editor_alias_id))

    # Watch the Researcher decide: it should search for the AWS question and not for the general one.
    for question in ["What does a Bedrock flow's execution role need to call a Lambda function?",
                     "What is a good name for a pet goldfish?"]:
        print(f"\nResearcher, directly: {question}")
        answer = flow_utils.invoke_agent(runtime, researcher_id, researcher_alias_id, question)
        print(f"  [answer] {answer}")

    # Now the same agent inside a flow, chained into the Editor.
    template = flow_utils.load_template(HERE / "multi_agent_flow.json", {
        "$$AGENT_1_ALIAS_ARN": researcher_arn,
        "$$AGENT_2_ALIAS_ARN": editor_arn,
    })
    flow_id = flow_utils.create_flow(
        agent, template["name"], template["description"], template["definition"], role_arn, template.get("tags"))
    version, alias_id = flow_utils.deploy_flow(agent, flow_id)

    question = args.question or "How do I invoke a Bedrock flow from my application?"
    outputs = flow_utils.invoke_flow(runtime, flow_id, alias_id, question)
    print(f"\nFlow (Researcher -> Editor): {question}\n{outputs[0]}")
finally:
    if args.keep and flow_id:
        print(f"\nKept flow {flow_id} (alias {alias_id}), agents {[a for a, _ in created]}, and Lambda {LAMBDA_NAME}.")
    else:
        flow_utils.cleanup_flow(agent, flow_id, alias_id, version)
        for agent_id, agent_alias_id in created:
            flow_utils.delete_agent(agent, agent_id, agent_alias_id)
        if lambda_arn:
            flow_utils.delete_lambda(lambda_client, LAMBDA_NAME)
    if args.delete_role:
        flow_utils.delete_role(iam)
        flow_utils.delete_agent_role(iam)
        flow_utils.delete_lambda_role(iam)
