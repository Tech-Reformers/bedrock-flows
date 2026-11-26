"""Lambda function: search AWS documentation through the AWS Knowledge MCP Server.

It is used two ways:
- Module 03, as a flow Lambda node. The node sends the user's question, and the function returns
  the top excerpts, each with its source URL, as one string for the "AWS expert" prompt.
- Module 05, as an agent's tool (an action group function named search_aws_docs). The agent
  decides when to call it and with what query.

Either way it calls the server's `aws___search_documentation` tool.

The AWS Knowledge MCP Server is public: it needs no AWS account or credentials, but it is
rate limited. MCP is JSON-RPC over HTTP, so the standard library is enough (no layer or
dependencies to package).

Flow Lambda node input event:
    {"messageVersion": "1.0", "flow": {...},
     "node": {"name": "...", "inputs": [{"name": "question", "type": "String", "value": "..."}]}}
Whatever the function returns becomes the node's `functionResponse` output.

Agent action group input event (function details):
    {"messageVersion": "1.0", "agent": {...}, "actionGroup": "...", "function": "search_aws_docs",
     "parameters": [{"name": "query", "type": "string", "value": "..."}], ...}
The agent expects a {"messageVersion", "response": {..., "functionResponse": ...}} reply.
"""
import json
import urllib.request

MCP_URL = "https://knowledge-mcp.global.api.aws"
MAX_RESULTS = 3
MAX_CHARS_PER_RESULT = 1500
NO_RESULTS = "No AWS documentation was found for this question."


def search_aws_docs(question):
    """Call the MCP tool and return a list of {title, url, context} results."""
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "aws___search_documentation",
            "arguments": {"search_phrase": question, "limit": MAX_RESULTS},
        },
    }).encode()
    request = urllib.request.Request(MCP_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    })
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read().decode()
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            # Streamable HTTP may answer as server-sent events: the JSON-RPC message is on a "data:" line.
            raw = next(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    result = json.loads(raw)["result"]
    if result.get("isError"):
        raise RuntimeError(result["content"][0].get("text", "MCP tool error"))
    # The tool's text content is itself JSON: {"content": {"result": [...]}}
    return json.loads(result["content"][0]["text"])["content"]["result"]


def format_results(query):
    """Search and format the excerpts as one string. Never raises."""
    try:
        results = search_aws_docs(query)
    except Exception as e:  # rate limit, timeout, or format change: let the model say it found nothing
        print(f"AWS docs search failed: {e}")
        return NO_RESULTS
    excerpts = [
        f"Source: {r.get('url', '')}\nTitle: {r.get('title', '')}\n{r.get('context', '')[:MAX_CHARS_PER_RESULT]}"
        for r in results
    ]
    return "\n\n---\n\n".join(excerpts) or NO_RESULTS


def lambda_handler(event, context):
    if "actionGroup" in event:
        return agent_action(event)
    # Flow Lambda node (module 03): the question arrives as a node input, the return value is the node output.
    inputs = {i["name"]: i.get("value") for i in event["node"]["inputs"]}
    return format_results(inputs.get("question") or "")


def agent_action(event):
    """Agent action group defined with function details (module 05).

    Input and response formats follow "Configure Lambda functions to send information that an
    Amazon Bedrock agent elicits from the user" in the Bedrock User Guide.
    """
    params = {p["name"]: p.get("value") for p in event.get("parameters", [])}
    query = params.get("query") or event.get("inputText", "")
    print(f"Agent {event['agent']['name']} called {event['function']}(query={query!r})")
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "function": event["function"],
            "functionResponse": {"responseBody": {"TEXT": {"body": format_results(query)}}},
        },
    }
