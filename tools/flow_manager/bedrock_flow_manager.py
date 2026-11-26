import boto3
import json
import sys as _sys
from pathlib import Path
from botocore.exceptions import ClientError
import logging
from typing import Tuple, Optional, Dict, List, Generator
import argparse
import sys
from termcolor import colored
import time
import os
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.json import JSON
from contextlib import contextmanager

# Shared IAM policy and knowledge base helpers live in the repo root
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (tools/flow_manager/ -> repo)
import flow_utils  # noqa: E402

# Configure logging
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Color scheme for better visibility in both dark and light modes
# Using high contrast colors that are also colorblind-friendly
COLORS = {
    'header': {'color': 'white', 'attrs': ['bold']},
    'step': {'color': 'cyan', 'attrs': ['bold']},
    'info': {'color': 'blue', 'attrs': ['bold']},
    'success': {'color': 'green', 'attrs': ['bold']},
    'warning': {'color': 'yellow', 'attrs': ['bold']},
    'error': {'color': 'red', 'attrs': ['bold']},
    'input': {'color': 'magenta', 'attrs': ['bold']},
}


def print_colored(message: str, style: str = 'info', prefix: str = ''):
    """Print colored message with consistent styling"""
    color = COLORS.get(style, COLORS['info'])
    print(colored(f"{prefix}{message}", color['color'], attrs=color.get('attrs', [])))


class FlowConversation:
    """Handles multi-turn conversations with Bedrock Flow"""

    def __init__(self, flow_id: str, alias_id: str, execution_id: str = None):
        self.flow_id = flow_id
        self.alias_id = alias_id
        self.execution_id = execution_id
        self.conversation_history = []

    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_formatted_history(self) -> str:
        """Get formatted conversation history"""
        formatted = "\nConversation History:\n" + "-" * 30 + "\n"
        for msg in self.conversation_history:
            formatted += f"[{msg['timestamp']}] {msg['role']}: {msg['content']}\n"
        return formatted


class BedrockFlowManager:
    def __init__(self, region: str, profile_name: Optional[str], existing_role_name: Optional[str] = None):
        """Initialize the BedrockFlowManager"""
        print_colored("\n=== Amazon Bedrock Flow Manager ===", 'header')
        print_colored(f"Region: {region}", 'info')
        print_colored(f"Profile: {profile_name or '(default credential chain)'}", 'info')

        self.region = region
        self.profile_name = profile_name
        self.session = boto3.Session(profile_name=profile_name)

        # Initialize AWS clients
        self.bedrock_client = self.session.client('bedrock-agent', region_name=region)
        self.bedrock_runtime = self.session.client('bedrock-agent-runtime', region_name=region)
        self.iam = self.session.client('iam')
        self.sts = self.session.client('sts')

        # Store role_arn after creation
        self.role_arn = self.create_iam_role(existing_role_name)

        self.console = Console()

    @contextmanager
    def flow_lifecycle(self, flow_id: Optional[str] = None, alias_id: Optional[str] = None,
                       version: Optional[str] = None) -> Generator:
        """Context manager to handle flow lifecycle and cleanup on errors"""
        created_resources = {
            'flow_id': flow_id,
            'alias_id': alias_id,
            'version': version
        }

        try:
            yield created_resources
        except BaseException:
            # BaseException so Ctrl+C also cleans up
            print_colored("\n🧹 Error occurred. Cleaning up resources...", 'warning')
            if any(created_resources.values()):
                self.cleanup_flow(
                    flow_id=created_resources.get('flow_id'),
                    alias_id=created_resources.get('alias_id'),
                    version=created_resources.get('version')
                )
            raise

    def cleanup_flow(self, flow_id: Optional[str], alias_id: Optional[str] = None,
                     version: Optional[str] = None) -> bool:
        """Clean up created flow resources. Returns True only if every deletion succeeded."""
        failed = []
        print_colored("\n🧹 Cleaning Up Resources", 'step')
        print_colored("-" * 30, 'info')

        try:
            if alias_id:
                print_colored("1. Deleting flow alias...", 'info')
                try:
                    self.bedrock_client.delete_flow_alias(
                        flowIdentifier=flow_id,
                        aliasIdentifier=alias_id
                    )
                    print_colored("   ✅ Alias deleted", 'success')
                except Exception as e:
                    failed.append(f"Alias ID: {alias_id}")
                    print_colored(f"   ⚠️  Error deleting alias: {str(e)}", 'warning')

            if version:
                print_colored("2. Deleting flow version...", 'info')
                try:
                    self.bedrock_client.delete_flow_version(
                        flowIdentifier=flow_id,
                        flowVersion=version
                    )
                    print_colored("   ✅ Version deleted", 'success')
                except Exception as e:
                    failed.append(f"Version: {version}")
                    print_colored(f"   ⚠️  Error deleting version: {str(e)}", 'warning')

            if flow_id:
                print_colored("3. Deleting flow...", 'info')
                try:
                    self.bedrock_client.delete_flow(flowIdentifier=flow_id)
                    print_colored("   ✅ Flow deleted", 'success')
                except Exception as e:
                    failed.append(f"Flow ID: {flow_id}")
                    print_colored(f"   ⚠️  Error deleting flow: {str(e)}", 'warning')

            if failed:
                print_colored("\n⚠️  Cleanup finished with errors. Delete these manually:", 'warning')
                for item in failed:
                    print_colored(f"  • {item}", 'warning')
                return False

            print_colored("\n✨ Cleanup completed", 'success')
            return True

        except Exception as e:
            print_colored(f"\n❌ Error during cleanup: {str(e)}", 'error')
            print_colored("Some resources may need to be cleaned up manually:", 'warning')
            if flow_id:
                print_colored(f"  • Flow ID: {flow_id}", 'warning')
            if alias_id:
                print_colored(f"  • Alias ID: {alias_id}", 'warning')
            if version:
                print_colored(f"  • Version: {version}", 'warning')
            return False

    @staticmethod
    def list_templates(templates_dir: str) -> List[Path]:
        """List flow templates (*_flow.json) found anywhere under the templates directory"""
        print_colored("\n📂 Available Templates:", 'step')
        print_colored("-" * 50, 'info')

        templates_path = Path(templates_dir)
        if not templates_path.exists():
            print_colored(f"Templates directory not found: {templates_dir}", 'error')
            return []

        skip = {'venv', '.venv', 'node_modules', '.git', '__pycache__'}
        templates = sorted(
            p for p in templates_path.rglob('*_flow.json')
            if not skip.intersection(p.relative_to(templates_path).parts)
        )

        if not templates:
            print_colored(f"No *_flow.json templates found under {templates_dir}.", 'warning')
            return []

        for idx, template in enumerate(templates, 1):
            # Try to get description from template
            try:
                with open(template, 'r') as f:
                    description = json.load(f).get('description', 'No description available')
            except (OSError, json.JSONDecodeError):
                description = 'Unable to read template description'

            print_colored(f"{idx}. {template.relative_to(templates_path)}", 'info')
            print_colored(f"   Description: {description}", 'info', prefix='   ')
            print_colored("-" * 50, 'info')

        return templates

    @staticmethod
    def select_template(templates: List[Path]) -> Optional[Path]:
        """Let user select a template interactively"""
        if not templates:
            return None

        while True:
            try:
                print_colored("\nSelect a template number:", 'input')
                choice = input(colored("Enter number (or 'q' to quit): ", COLORS['input']['color']))

                if choice.lower() == 'q':
                    sys.exit(0)

                idx = int(choice) - 1
                if 0 <= idx < len(templates):
                    return templates[idx]
                else:
                    print_colored("Invalid selection! Please try again.", 'error')
            except ValueError:
                print_colored("Please enter a valid number!", 'error')

    def _build_trust_policy(self, account_id: str) -> dict:
        """Trust policy limited to Bedrock flows in this account and region (confused-deputy protection)"""
        return flow_utils.build_trust_policy(self.region, account_id)

    def _build_permissions_policy(self, account_id: str) -> dict:
        """Runtime permissions for flow nodes, scoped to this account/region where the ARN allows"""
        return flow_utils.build_permissions_policy(self.region, account_id)

    def create_iam_role(self, existing_role_name: Optional[str] = None) -> str:
        """Create IAM role for Bedrock Flows or use existing role"""
        
        # If existing role name is provided, use it without printing step message
        if existing_role_name:
            try:
                role = self.iam.get_role(RoleName=existing_role_name)
                print_colored(f"Using existing IAM role: {existing_role_name}", 'info')
                return role['Role']['Arn']
            except ClientError as e:
                print_colored(f"❌ Error getting existing role: {str(e)}", 'error')
                raise e

        try:
            # Try to get existing default role first
            role = self.iam.get_role(RoleName=flow_utils.ROLE_NAME)
            print_colored(f"Using existing IAM role: {flow_utils.ROLE_NAME}", 'info')
            return role['Role']['Arn']
            
        except ClientError as e:
            if e.response['Error']['Code'] != 'NoSuchEntity':
                raise e
                
            # Only print step message when actually creating a new role
            print_colored("\n🔑 Step 1: Setting up IAM Role", 'step')
            print_colored("-" * 30, 'info')

            account_id = self.sts.get_caller_identity()['Account']
            trust_policy = self._build_trust_policy(account_id)
            bedrock_policy = self._build_permissions_policy(account_id)

            print_colored(f"Creating IAM role: {flow_utils.ROLE_NAME}...", 'info')
            response = self.iam.create_role(
                RoleName=flow_utils.ROLE_NAME,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Role for Amazon Bedrock Flows'
            )

            # Create and attach the custom policy
            print_colored("Creating and attaching Bedrock policy...", 'info')
            policy_name = 'BedrockFlowsPolicy'
            self.iam.put_role_policy(
                RoleName=flow_utils.ROLE_NAME,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(bedrock_policy)
            )

            print_colored("✅ IAM role created successfully", 'success')
            return response['Role']['Arn']
    
    VARIABLE_PATTERN = re.compile(r'\$\$[A-Za-z][A-Za-z0-9_]*')

    @classmethod
    def _collect_variables(cls, node, found: set) -> None:
        """Collect $$VARIABLE names from all string values in a parsed JSON structure"""
        if isinstance(node, str):
            found.update(cls.VARIABLE_PATTERN.findall(node))
        elif isinstance(node, dict):
            for value in node.values():
                cls._collect_variables(value, found)
        elif isinstance(node, list):
            for value in node:
                cls._collect_variables(value, found)

    @classmethod
    def _substitute_variables(cls, node, replacements: Dict[str, str]):
        """Return a copy of a parsed JSON structure with $$VARIABLEs replaced.

        Matches whole identifiers, so $$KB never clobbers $$KB_ID, and values
        are inserted as plain strings (no JSON escaping issues).
        """
        if isinstance(node, str):
            return cls.VARIABLE_PATTERN.sub(lambda m: replacements.get(m.group(0), m.group(0)), node)
        if isinstance(node, dict):
            return {k: cls._substitute_variables(v, replacements) for k, v in node.items()}
        if isinstance(node, list):
            return [cls._substitute_variables(v, replacements) for v in node]
        return node

    def find_knowledge_base_id(self, name: str) -> Optional[str]:
        """Look up a Bedrock knowledge base ID by exact name (None if missing, inactive, or managed)"""
        return flow_utils.find_knowledge_base_id(self.bedrock_client, name)

    def process_template(self, template_path: Path, knowledge_base_name: Optional[str] = None) -> Tuple[dict, bool, dict]:
        """Process template and replace variables"""
        print_colored("\n📝 Step 2: Processing Template", 'step')
        print_colored("-" * 30, 'info')

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        print_colored(f"Loading template: {template_path.name}", 'info')
        with open(template_path, 'r') as f:
            template = json.load(f)

        # Validate template structure
        required_fields = ['definition', 'description', 'name']
        missing_fields = [field for field in required_fields if field not in template]
        if missing_fields:
            raise ValueError(f"Template missing required fields: {', '.join(missing_fields)}")

        # Extract metadata
        template_metadata = {
            'description': template.get('description', ''),
            'name': template.get('name', ''),
            'tags': template.get('tags', {}),
            'executionRoleArn': template.get('executionRoleArn', None)
        }

        is_iterator = 'iterator' in template_path.stem.lower()

        # Find variables ($$NAME) in the string values of the definition and metadata
        definition = template['definition']
        variables = set()
        self._collect_variables(definition, variables)
        self._collect_variables(template_metadata, variables)

        if not variables:
            print_colored("No variables found in template", 'info')
            return definition, is_iterator, template_metadata

        print_colored("\nTemplate Variables Found:", 'warning')
        print_colored("-" * 30, 'info')

        # Sort variables for consistent display
        sorted_variables = sorted(variables, key=lambda x: (x.lower(), x))
        for var in sorted_variables:
            print_colored(f"• {var}", 'info')

        # Interactive replacement
        replacements = {}
        print_colored("\n🔄 Variable Replacement", 'step')
        print_colored("Enter values for each variable:", 'info')

        # Optional: fill the knowledge base ID by looking the knowledge base up by name
        if knowledge_base_name and '$$KNOWLEDGEBASE_ID' in variables:
            kb_id = self.find_knowledge_base_id(knowledge_base_name)
            if kb_id:
                replacements['$$KNOWLEDGEBASE_ID'] = kb_id
                print_colored(f"Found knowledge base '{knowledge_base_name}': {kb_id}", 'success')
            else:
                print_colored(f"No usable knowledge base named '{knowledge_base_name}'; enter the ID manually.", 'warning')

        for var in sorted_variables:
            if var in replacements:
                continue
            while True:
                value = input(colored(f"Enter value for {var}: ", COLORS['input']['color'])).strip()
                if value:
                    replacements[var] = value
                    break
                print_colored("Value cannot be empty! Please try again.", 'error')

        # Substitute into the parsed structure, so values are never interpreted as JSON
        definition = self._substitute_variables(definition, replacements)
        template_metadata = self._substitute_variables(template_metadata, replacements)

        print_colored("\n✅ Template processing complete", 'success')
        print_colored("\nTemplate Metadata:", 'info')
        print_colored(f"  • Name: {template_metadata['name']}", 'info')
        print_colored(f"  • Description: {template_metadata['description']}", 'info')
        if template_metadata['tags']:
            print_colored("  • Tags:", 'info')
            for key, value in template_metadata['tags'].items():
                print_colored(f"    - {key}: {value}", 'info')

        return definition, is_iterator, template_metadata

    def _create_flow_with_retry(self, create_args: dict, attempts: int = 6, delay: int = 5) -> dict:
        """create_flow, retrying while a freshly created IAM role propagates"""
        for attempt in range(1, attempts + 1):
            try:
                return self.bedrock_client.create_flow(**create_args)
            except ClientError as e:
                code = e.response['Error']['Code']
                message = e.response['Error'].get('Message', '').lower()
                role_not_ready = code == 'ValidationException' and 'role' in message
                if not role_not_ready or attempt == attempts:
                    raise
                print_colored(
                    f"IAM role not usable yet, retrying in {delay}s ({attempt}/{attempts - 1})...", 'warning')
                time.sleep(delay)

    def create_flow(self, flow_definition: dict, template_metadata: dict, flow_name: str = None) -> str:
        """Create a Bedrock Flow from definition"""
        print_colored("\n🚀 Step 3: Creating Flow", 'step')
        print_colored("-" * 30, 'info')

        try:
            # Use template name if no flow name provided
            final_flow_name = flow_name or template_metadata['name']
            print_colored(f"Creating flow: {final_flow_name}", 'info')
            if not flow_name:
                print_colored("Using flow name from template", 'info')

            print_colored("Processing flow definition...", 'info')

            # Prepare create_flow arguments
            create_args = {
                'name': final_flow_name,
                'description': template_metadata['description'],
                'definition': flow_definition,
                'executionRoleArn': template_metadata.get('executionRoleArn') or self.role_arn,
            }

            # Add tags if present
            if template_metadata.get('tags'):
                create_args['tags'] = template_metadata['tags']

            # Create flow
            response = self._create_flow_with_retry(create_args)

            flow_id = response['id']
            print_colored("✅ Flow created successfully!", 'success')
            print_colored(f"Flow ID: {flow_id}", 'info')
            print_colored(f"Flow Name: {final_flow_name}", 'info')

            return flow_id

        except Exception as e:
            print_colored(f"❌ Error creating flow: {str(e)}", 'error')
            raise e

    def _wait_for_flow_prepared(self, flow_id: str, timeout: int = 300, interval: int = 3):
        """Poll until the flow status is Prepared (prepare_flow is asynchronous)"""
        deadline = time.time() + timeout
        while True:
            status = self.bedrock_client.get_flow(flowIdentifier=flow_id)['status']
            if status == 'Prepared':
                return
            if status == 'Failed':
                raise RuntimeError(f"Flow {flow_id} failed to prepare")
            if time.time() >= deadline:
                raise TimeoutError(f"Flow {flow_id} not prepared after {timeout}s (status: {status})")
            time.sleep(interval)

    def prepare_flow(self, flow_id: str, resources: Optional[dict] = None) -> Tuple[str, str]:
        """Prepare flow for execution"""
        print_colored("\n⚙️ Step 4: Preparing Flow", 'step')
        print_colored("-" * 30, 'info')

        try:
            # Prepare flow
            print_colored("Preparing flow...", 'info')
            self.bedrock_client.prepare_flow(flowIdentifier=flow_id)
            self._wait_for_flow_prepared(flow_id)

            # Create version
            print_colored("Creating flow version...", 'info')
            version_response = self.bedrock_client.create_flow_version(flowIdentifier=flow_id)
            flow_version = version_response['version']
            if resources is not None:
                resources['version'] = flow_version  # recorded now so cleanup sees it if alias creation fails
            print_colored(f"Created version: {flow_version}", 'success')

            # Create alias
            print_colored("Creating flow alias...", 'info')
            alias_response = self.bedrock_client.create_flow_alias(
                flowIdentifier=flow_id,
                name='latest',
                description=f"Alias for version {flow_version}",
                routingConfiguration=[{'flowVersion': flow_version}]
            )

            alias_id = alias_response['id']
            if resources is not None:
                resources['alias_id'] = alias_id

            print_colored("\n✅ Flow preparation complete!", 'success')
            print_colored("Flow Details:", 'info')
            print_colored(f"  • Version: {flow_version}", 'info')
            print_colored(f"  • Alias ID: {alias_id}", 'info')

            return flow_version, alias_id

        except Exception as e:
            print_colored(f"❌ Error preparing flow: {str(e)}", 'error')
            raise e

    def format_flow_response(self, response):
        """Format flow response for better display"""
        try:
            if not response:
                print_colored("\n⚠️  No response received", 'warning')
                return

            print_colored("\n📊 Flow Response:", 'step')
            print_colored("-" * 30, 'info')

            # Handle list responses (from iterator templates)
            if isinstance(response, list):
                print_colored("Iterator Response:", 'info')
                for i, item in enumerate(response, 1):
                    # Try to detect and format each item
                    print_colored(f"\n[Response {i}]", 'step')
                    print_colored("-" * 20, 'info')

                    # Remove markdown code block indicators if present
                    if isinstance(item, str):
                        item = item.replace('```json', '').replace('```', '').strip()

                    try:
                        # Try to parse as JSON
                        json_data = json.loads(item) if isinstance(item, str) else item
                        self.console.print(Panel(
                            JSON(json.dumps(json_data, indent=2)),
                            title=f"JSON Response {i}",
                            border_style="cyan"
                        ))
                    except (json.JSONDecodeError, TypeError):
                        # If not JSON, check for markdown
                        if isinstance(item, str) and any(md_char in item for md_char in ['#', '```', '**', '_', '>']):
                            try:
                                self.console.print(Panel(
                                    Markdown(item),
                                    title=f"Markdown Response {i}",
                                    border_style="cyan"
                                ))
                            except Exception:
                                # If markdown parsing fails, display as text
                                self.console.print(Panel(
                                    item,
                                    title=f"Text Response {i}",
                                    border_style="cyan"
                                ))
                        else:
                            # Display as plain text
                            self.console.print(Panel(
                                str(item),
                                title=f"Text Response {i}",
                                border_style="cyan"
                            ))
                return

            # Handle single response
            # Remove markdown code block indicators if present
            if isinstance(response, str):
                response = response.replace('```json', '').replace('```', '').strip()

            try:
                # Try to parse as JSON
                json_data = json.loads(response) if isinstance(response, str) else response
                self.console.print(Panel(
                    JSON(json.dumps(json_data, indent=2)),
                    title="JSON Response",
                    border_style="cyan"
                ))
                return
            except (json.JSONDecodeError, TypeError):
                # If not JSON, check for markdown
                if isinstance(response, str) and any(md_char in response for md_char in ['#', '```', '**', '_', '>']):
                    try:
                        self.console.print(Panel(
                            Markdown(response),
                            title="Markdown Response",
                            border_style="cyan"
                        ))
                        return
                    except Exception as e:
                        print_colored(f"\n⚠️  Error formatting response: {str(e)}", 'error')


                # If it's a multi-line response
                if isinstance(response, str) and '\n' in response:
                    self.console.print(Panel(
                        response,
                        title="Multi-line Response",
                        border_style="cyan"
                    ))
                    return

                # For simple text responses
                self.console.print(Panel(
                    str(response),
                    title="Text Response",
                    border_style="cyan"
                ))

        except Exception as e:
            # Fallback for any unexpected errors
            print_colored(f"\n⚠️  Error formatting response: {str(e)}", 'error')
            print_colored("Displaying raw response:", 'info')
            print(response)

    def test_flow(self, flow_id: str, alias_id: str, input_text: str | list, is_iterator: bool = False) -> str:
        """Test the created flow with multi-turn support"""
        print_colored("\n🧪 Step 5: Testing Flow", 'step')
        print_colored("-" * 30, 'info')

        # Initialize conversation
        conversation = FlowConversation(flow_id, alias_id)

        try:
            while True:
                # Prepare input payload
                input_payload = self._prepare_input_payload(
                    input_text,
                    is_iterator,
                    conversation.execution_id
                )

                print_colored("\nInvoking flow...", 'warning')
                start_time = time.time()

                # Invoke flow
                response = self.bedrock_runtime.invoke_flow(
                    flowIdentifier=flow_id,
                    flowAliasIdentifier=alias_id,
                    **({"executionId": conversation.execution_id} if conversation.execution_id else {}),
                    inputs=[input_payload]
                )

                # Process response stream
                result = self._process_response_stream(response, conversation)
                execution_time = time.time() - start_time

                # Handle completion
                if result['status'] == 'SUCCESS':
                    print_colored(f"\n✅ Flow execution successful! ({execution_time:.2f}s)", 'success')
                    self.format_flow_response(result['output'])
                    return result['output']

                # Handle multi-turn request
                elif result['status'] == 'INPUT_REQUIRED':
                    print_colored("\n👥 Additional input required:", 'info')
                    self.format_flow_response(result['prompt'])

                    # Get user input
                    input_text = input(colored("\nYour response: ", COLORS['input']['color']))
                    conversation.add_to_history('user', input_text)

                    # Update node information for next turn
                    input_text = {
                        'text': input_text,
                        'node_name': result['node_name'],
                        'is_initial': False
                    }
                else:
                    raise Exception(f"Unexpected flow status: {result['status']}")

        except Exception as e:
            print_colored(f"❌ Error testing flow: {str(e)}", 'error')
            raise e
        
    def export_flow_definition(self, flow_id: str, output_path: str = None) -> dict:
        """
        Export flow definition to a JSON file
        
        Args:
            flow_id (str): The ID of the flow to export
            output_path (str, optional): Path to save the JSON file. If None, uses flow name
            
        Returns:
            dict: The flow definition
        """
        print_colored("\n📤 Exporting Flow Definition", 'step')
        print_colored("-" * 30, 'info')
        
        try:
            # Get flow details
            flow_details = self.bedrock_client.get_flow(flowIdentifier=flow_id)
            
            # Extract relevant information
            flow_data = {
                "name": flow_details['name'],
                "description": flow_details.get('description', ''),
                "definition": flow_details['definition'],
                "tags": flow_details.get('tags', {}),
                "executionRoleArn": flow_details.get('executionRoleArn')
            }
            
            # Generate output path if not provided
            if not output_path:
                flow_name = flow_details['name'].lower().replace(' ', '_')
                output_path = f"./{flow_name}_exported.json"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(flow_data, f, indent=2)
                
            print_colored(f"✅ Flow definition exported successfully to: {output_path}", 'success')
            print_colored("\nExported Flow Details:", 'info')
            print_colored(f"  • Name: {flow_data['name']}", 'info')
            print_colored(f"  • Description: {flow_data['description']}", 'info')
            if flow_data['tags']:
                print_colored("  • Tags:", 'info')
                for key, value in flow_data['tags'].items():
                    print_colored(f"    - {key}: {value}", 'info')
            
            return flow_data
            
        except Exception as e:
            print_colored(f"❌ Error exporting flow definition: {str(e)}", 'error')
            raise e

    def _prepare_input_payload(self, input_data: str | dict, is_iterator: bool, execution_id: str = None) -> dict:
        """Prepare input payload for flow invocation"""

        # Handle dictionary input for multi-turn
        if isinstance(input_data, dict):
            node_name = input_data['node_name']
            is_initial = input_data['is_initial']
            content = input_data['text']
        else:
            node_name = "FlowInputNode"
            is_initial = True
            content = input_data

        # Prepare content based on iterator status
        if is_iterator:
            if isinstance(content, str):
                content = [content]
            elif not isinstance(content, list):
                raise ValueError(f"Unsupported input type: {type(content)}")

        payload = {
            "content": {"document": content},
            "nodeName": node_name
        }

        # Add appropriate node name based on turn
        if is_initial:
            payload["nodeOutputName"] = "document"
        else:
            payload["nodeInputName"] = "agentInputText"

        return payload

    def _process_response_stream(self, response: dict, conversation: FlowConversation) -> dict:
        """Process response stream from flow invocation"""

        result = {
            'status': None,
            'output': None,
            'prompt': None,
            'node_name': None
        }

        # Update execution ID
        conversation.execution_id = response.get('executionId', conversation.execution_id)

        # Process stream events
        for event in response.get("responseStream", []):
            if 'flowCompletionEvent' in event:
                result['status'] = event['flowCompletionEvent']['completionReason']

            elif 'flowMultiTurnInputRequestEvent' in event:
                result['prompt'] = event['flowMultiTurnInputRequestEvent']['content']['document']
                result['node_name'] = event['flowMultiTurnInputRequestEvent']['nodeName']
                conversation.add_to_history('assistant', result['prompt'])

            elif 'flowOutputEvent' in event:
                result['output'] = event['flowOutputEvent']['content']['document']
                conversation.add_to_history('assistant', result['output'])

        return result

def get_default_profile() -> Optional[str]:
    """Profile from the environment, or None so boto3 uses its default credential chain
    (env vars, SSO, instance/task roles) instead of requiring a 'default' profile."""
    return os.environ.get('AWS_PROFILE') or os.environ.get('AWS_DEFAULT_PROFILE')


def resolve_region(region: Optional[str], profile: Optional[str]) -> str:
    """Resolve the region: --region, then env vars, then the selected profile's config."""
    region = region or \
             os.environ.get('AWS_REGION') or \
             os.environ.get('AWS_DEFAULT_REGION') or \
             boto3.Session(profile_name=profile).region_name

    if not region:
        raise SystemExit(
            "No AWS region found. Pass --region, set AWS_REGION, "
            "or configure one for your profile (aws configure set region <region>)."
        )

    return region


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create and run Amazon Bedrock Flow from template',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Get defaults
    default_profile = get_default_profile()

    parser.add_argument(
        '--region',
        default=None,
        help='AWS region (default: AWS_REGION/AWS_DEFAULT_REGION, then the profile\'s configured region)'
    )
    parser.add_argument(
        '--profile',
        default=default_profile,
        help=f'AWS profile name (default: {default_profile or "default credential chain"})'
    )
    parser.add_argument(
        '--flow-name',
        help='Optional: Override flow name from template'
    )
    parser.add_argument(
        '--test-input',
        nargs='+',  # This allows multiple inputs
        help='Input text(s) for testing the flow. For single input, provide one string. For multiple inputs, provide space-separated strings'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up resources after testing'
    )
    parser.add_argument(
        '--templates-dir',
        default=str(Path(__file__).resolve().parents[2]),
        help='Directory searched for *_flow.json templates (default: the repo root)'
    )

    parser.add_argument(
        '--knowledge-base-name',
        default=os.environ.get('BEDROCK_KB_NAME'),
        help='Optional: name of a Bedrock knowledge base to use for $$KNOWLEDGEBASE_ID '
             '(default: BEDROCK_KB_NAME env var). If not found, you are prompted for the ID.'
    )

    parser.add_argument(
        '--existing-role',
        help='Name of existing IAM role to use instead of creating a new one'
    )

    args = parser.parse_args()
    args.region = resolve_region(args.region, args.profile)

    # Convert test_input to appropriate format if provided
    if args.test_input:
        if len(args.test_input) == 1:
            args.test_input = args.test_input[0]  # Single string
        else:
            args.test_input = list(args.test_input)  # List of strings

    # Print configuration
    print_colored("\n📋 Configuration:", 'header')
    print_colored(f"Region: {args.region}", 'info')
    print_colored(f"Profile: {args.profile or '(default credential chain)'}", 'info')
    print_colored(f"Templates Directory: {args.templates_dir}", 'info')
    if args.test_input:
        print_colored("Test Input:", 'info')
        if isinstance(args.test_input, list):
            for i, input_text in enumerate(args.test_input, 1):
                print_colored(f"  {i}. {input_text}", 'info')
        else:
            print_colored(f"  • {args.test_input}", 'info')

    return args


def main():
    args = parse_args()

    try:
        # Initialize flow manager
        flow_manager = BedrockFlowManager(args.region, args.profile, args.existing_role)

        # List and select template
        templates = BedrockFlowManager.list_templates(args.templates_dir)
        if not templates:
            sys.exit(1)

        selected_template = BedrockFlowManager.select_template(templates)
        if not selected_template:
            sys.exit(1)

        # Process template and replace variables
        flow_definition, is_iterator, template_metadata = flow_manager.process_template(
            selected_template, args.knowledge_base_name)

        # Use context manager for flow lifecycle
        with flow_manager.flow_lifecycle() as resources:
            # Create flow
            flow_id = flow_manager.create_flow(
                flow_definition=flow_definition,
                template_metadata=template_metadata,
                flow_name=args.flow_name
            )
            resources['flow_id'] = flow_id
            # Prepare flow
            version, alias_id = flow_manager.prepare_flow(flow_id, resources)

            # Only prepare flow and test if test input is provided
            if args.test_input:
                # Test flow with multi-turn support
                flow_manager.test_flow(
                    flow_id,
                    alias_id,
                    args.test_input,
                    is_iterator
                )

            # If cleanup flag is set, cleanup resources
            if args.cleanup:
                cleaned = flow_manager.cleanup_flow(
                    flow_id,
                    resources.get('alias_id'),
                    resources.get('version')
                )
                if not cleaned:
                    print_colored("\n❌ Completed, but cleanup was incomplete.", 'error')
                    sys.exit(1)

    except Exception as e:
        print_colored(f"\n❌ Error: {str(e)}", 'error')
        sys.exit(1)

    print_colored("\n✨ Operation completed successfully!", 'success')


if __name__ == "__main__":
    main()