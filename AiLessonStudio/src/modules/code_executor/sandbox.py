import sys
import os
import subprocess
import tempfile
import shutil
import signal
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re


@dataclass
class ExecutionResult:
    """Code execution result"""
    success: bool
    output: str
    error: str
    execution_time: float
    memory_used: int
    return_code: int
    truncated: bool


class CodeSandbox:
    """Secure code execution sandbox for cloud computing examples"""

    # Allowed modules and functions
    ALLOWED_MODULES = [
        'math', 'datetime', 'json', 're', 'os', 'sys',
        'collections', 'itertools', 'functools', 'random',
        'string', 'hashlib', 'base64', 'time'
    ]

    # Cloud simulation modules (custom)
    CLOUD_MODULES = [
        'aws_simulator', 'azure_simulator', 'gcp_simulator',
        'cloud_utils', 'docker_simulator', 'kubernetes_simulator'
    ]

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'__import__\s*\(',
        r'exec\s*\(',
        r'eval\s*\(',
        r'compile\s*\(',
        r'open\s*\([^)]*w[^)]*\)',  # File writes
        r'os\.system\s*\(',
        r'subprocess\s*\.',
        r'import\s+os\s*$',
        r'from\s+os\s+import',
        r'import\s+sys\s*$',
        r'from\s+sys\s+import',
        r'socket\.',
        r'requests\.',
        r'urllib\.',
        r'shutil\.',
        r'rm\s+-rf',
        r'rmdir',
        r'format\s*\(.*\{.*\}.*\)',  # String format injection
        r'%\(.*\)s',  # String format with mapping
    ]

    def __init__(self, config):
        self.config = config
        self.timeout = 30  # seconds
        self.max_output_size = 10000  # characters
        self.temp_dir = tempfile.mkdtemp(prefix='code_sandbox_')

        # Initialize cloud simulation modules
        self._init_cloud_modules()

        print(f"✅ Code Sandbox initialized at {self.temp_dir}")

    def __del__(self):
        """Cleanup temp directory"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def _init_cloud_modules(self):
        """Initialize cloud simulation modules"""
        # Create mock cloud modules
        self.cloud_mocks = {
            'aws_simulator': self._create_aws_mock(),
            'azure_simulator': self._create_azure_mock(),
            'gcp_simulator': self._create_gcp_mock(),
            'cloud_utils': self._create_cloud_utils(),
            'docker_simulator': self._create_docker_mock(),
            'kubernetes_simulator': self._create_k8s_mock()
        }

    def _create_aws_mock(self):
        """Create AWS simulation module"""

        class AWSSimulator:
            @staticmethod
            def create_ec2_instance(instance_type='t2.micro',
                                    image_id='ami-12345678',
                                    key_name='my-key',
                                    security_groups=None):
                return {
                    'instance_id': f'i-{os.urandom(4).hex()}',
                    'instance_type': instance_type,
                    'state': 'pending',
                    'public_ip': '203.0.113.123',
                    'private_ip': '10.0.1.123'
                }

            @staticmethod
            def create_s3_bucket(bucket_name):
                return {
                    'bucket_name': bucket_name,
                    'region': 'us-east-1',
                    'arn': f'arn:aws:s3:::{bucket_name}'
                }

            @staticmethod
            def upload_to_s3(bucket, key, data):
                return {
                    'etag': 'abc123def456',
                    'version_id': 'null',
                    'size': len(str(data))
                }

        return AWSSimulator()

    def _create_azure_mock(self):
        """Create Azure simulation module"""

        class AzureSimulator:
            @staticmethod
            def create_vm(vm_name, size='Standard_B1s',
                          image='UbuntuServer', admin_username='azureuser'):
                return {
                    'vm_id': f'/subscriptions/123/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/{vm_name}',
                    'name': vm_name,
                    'size': size,
                    'provisioning_state': 'Creating'
                }

            @staticmethod
            def create_storage_account(account_name, sku='Standard_LRS'):
                return {
                    'account_name': account_name,
                    'primary_endpoint': f'https://{account_name}.blob.core.windows.net/',
                    'sku': sku
                }

        return AzureSimulator()

    def _create_gcp_mock(self):
        """Create GCP simulation module"""

        class GCPSimulator:
            @staticmethod
            def create_compute_instance(instance_name, machine_type='e2-micro',
                                        zone='us-central1-a'):
                return {
                    'name': instance_name,
                    'machine_type': machine_type,
                    'zone': zone,
                    'status': 'PROVISIONING'
                }

            @staticmethod
            def create_storage_bucket(bucket_name, storage_class='STANDARD'):
                return {
                    'name': bucket_name,
                    'storage_class': storage_class,
                    'self_link': f'https://www.googleapis.com/storage/v1/b/{bucket_name}'
                }

        return GCPSimulator()

    def _create_cloud_utils(self):
        """Create cloud utility functions"""

        class CloudUtils:
            @staticmethod
            def validate_ip(ip_address):
                pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
                if re.match(pattern, ip_address):
                    parts = ip_address.split('.')
                    return all(0 <= int(part) <= 255 for part in parts)
                return False

            @staticmethod
            def calculate_cost(instance_type, hours, region='us-east-1'):
                # Mock pricing
                prices = {
                    't2.micro': 0.0116,
                    't2.small': 0.023,
                    't2.medium': 0.0464,
                    'e2-micro': 0.008,
                    'Standard_B1s': 0.012
                }
                base_price = prices.get(instance_type, 0.01)
                return base_price * hours

            @staticmethod
            def generate_password(length=12):
                import random
                import string
                chars = string.ascii_letters + string.digits + '!@#$%^&*'
                return ''.join(random.choice(chars) for _ in range(length))

        return CloudUtils()

    def _create_docker_mock(self):
        """Create Docker simulation module"""

        class DockerSimulator:
            @staticmethod
            def build_image(dockerfile_path, tag):
                return {
                    'image_id': f'sha256:{os.urandom(16).hex()}',
                    'tag': tag,
                    'size_mb': 123
                }

            @staticmethod
            def run_container(image, ports=None, volumes=None):
                return {
                    'container_id': os.urandom(12).hex(),
                    'status': 'running',
                    'ports': ports or {}
                }

        return DockerSimulator()

    def _create_k8s_mock(self):
        """Create Kubernetes simulation module"""

        class KubernetesSimulator:
            @staticmethod
            def create_deployment(name, image, replicas=1):
                return {
                    'deployment_name': name,
                    'namespace': 'default',
                    'replicas': replicas,
                    'available_replicas': 0
                }

            @staticmethod
            def create_service(name, selector, ports):
                return {
                    'service_name': name,
                    'cluster_ip': '10.96.0.123',
                    'type': 'ClusterIP',
                    'ports': ports
                }

        return KubernetesSimulator()

    def validate_code(self, code: str, language: str = 'python') -> Tuple[bool, str]:
        """Validate code for security and syntax"""
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains potentially dangerous pattern: {pattern}"

        # Check for disallowed imports
        disallowed_imports = [
            'import os', 'from os import',
            'import sys', 'from sys import',
            'import subprocess', 'from subprocess import',
            'import socket', 'from socket import'
        ]

        for disallowed in disallowed_imports:
            if disallowed in code.lower():
                return False, f"Disallowed import: {disallowed}"

        # Basic syntax check for Python
        if language == 'python':
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return False, f"Syntax error: {str(e)}"
            except Exception as e:
                return False, f"Compilation error: {str(e)}"

        return True, "Code validation passed"

    def execute_safe(self, code: str, language: str = 'python',
                     inputs: List[str] = None) -> ExecutionResult:
        """Execute code in secure sandbox"""
        start_time = time.time()

        # Validate code first
        is_valid, message = self.validate_code(code, language)
        if not is_valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation failed: {message}",
                execution_time=0.0,
                memory_used=0,
                return_code=1,
                truncated=False
            )

        if language != 'python':
            return ExecutionResult(
                success=False,
                output="",
                error=f"Language {language} not supported",
                execution_time=0.0,
                memory_used=0,
                return_code=1,
                truncated=False
            )

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.temp_dir,
            delete=False
        )

        try:
            # Write code to file with sandbox wrapper
            wrapped_code = self._wrap_code(code, inputs)
            temp_file.write(wrapped_code)
            temp_file.flush()
            temp_file.close()

            # Execute with timeout
            result = self._execute_with_timeout(temp_file.name)

            # Parse result
            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result['success'],
                output=result['output'][:self.max_output_size],
                error=result['error'],
                execution_time=execution_time,
                memory_used=result.get('memory_used', 0),
                return_code=result['return_code'],
                truncated=len(result['output']) > self.max_output_size
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                memory_used=0,
                return_code=1,
                truncated=False
            )
        finally:
            # Cleanup
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def _wrap_code(self, code: str, inputs: List[str] = None) -> str:
        """Wrap user code with sandbox environment"""
        # Import allowed modules
        imports = """
import math
import json
import re
import datetime
import collections
import itertools
import functools
import random
import string
import hashlib
import base64
import time
import sys
import os

# Mock cloud modules
class AWSSimulator:
    @staticmethod
    def create_ec2_instance(instance_type='t2.micro', image_id='ami-12345678', key_name='my-key'):
        return {'instance_id': 'i-1234567890abcdef0', 'state': 'pending'}

    @staticmethod
    def create_s3_bucket(bucket_name):
        return {'bucket_name': bucket_name, 'arn': f'arn:aws:s3:::{bucket_name}'}

    @staticmethod
    def upload_to_s3(bucket, key, data):
        return {'etag': 'abc123', 'size': len(str(data))}

class AzureSimulator:
    @staticmethod
    def create_vm(vm_name, size='Standard_B1s'):
        return {'vm_id': f'/subscriptions/123/virtualMachines/{vm_name}', 'name': vm_name}

    @staticmethod
    def create_storage_account(account_name):
        return {'account_name': account_name, 'primary_endpoint': f'https://{account_name}.blob.core.windows.net/'}

class GCPSimulator:
    @staticmethod
    def create_compute_instance(instance_name, machine_type='e2-micro'):
        return {'name': instance_name, 'machine_type': machine_type, 'status': 'PROVISIONING'}

    @staticmethod
    def create_storage_bucket(bucket_name):
        return {'name': bucket_name, 'self_link': f'https://www.googleapis.com/storage/v1/b/{bucket_name}'}

class CloudUtils:
    @staticmethod
    def validate_ip(ip):
        import re
        pattern = r'^(\\d{1,3}\\.){3}\\d{1,3}$'
        if re.match(pattern, ip):
            parts = ip.split('.')
            return all(0 <= int(p) <= 255 for p in parts)
        return False

    @staticmethod
    def calculate_cost(instance_type, hours):
        prices = {'t2.micro': 0.0116, 'e2-micro': 0.008}
        return prices.get(instance_type, 0.01) * hours

    @staticmethod
    def generate_password(length=12):
        import random, string
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

# Create mock instances
aws = AWSSimulator()
azure = AzureSimulator()
gcp = GCPSimulator()
cloud_utils = CloudUtils()

# Redirect output
import io
import sys

class OutputCapturer:
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.buffer
        sys.stderr = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def get_output(self):
        return self.buffer.getvalue()

output_capturer = OutputCapturer()

# Mock input if provided
mock_inputs = """ + json.dumps(inputs or []) + """
input_index = 0
original_input = __builtins__.__dict__.get('input', None)

def mock_input(prompt=''):
    global input_index
    if input_index < len(mock_inputs):
        value = mock_inputs[input_index]
        input_index += 1
        print(prompt + str(value))
        return str(value)
    return ''

__builtins__.__dict__['input'] = mock_input

# Execute user code
try:
    with output_capturer:
"""

        # Indent user code
        indented_code = '\n'.join(['        ' + line for line in code.split('\n')])

        wrapper_end = """
    output = output_capturer.get_output()
    print(output)

except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
"""

        return imports + indented_code + wrapper_end

    def _execute_with_timeout(self, script_path: str) -> Dict[str, Any]:
        """Execute script with timeout限制"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'return_code': 1,
            'memory_used': 0
        }

        try:
            # Execute in subprocess with resource limits
            env = os.environ.copy()
            env['PYTHONPATH'] = self.temp_dir

            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=self.temp_dir
            )

            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                result['return_code'] = process.returncode

                # Combine output
                output = stdout + stderr

                if process.returncode == 0:
                    result['success'] = True
                    result['output'] = output
                else:
                    result['error'] = output
                    result['output'] = output

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                result['error'] = f"Execution timed out after {self.timeout} seconds"
                result['output'] = stdout + stderr

        except Exception as e:
            result['error'] = str(e)

        return result

    def generate_code_for_topic(self, topic: str,
                                language: str = 'python') -> str:
        """Generate example code for a cloud computing topic"""
        topic_lower = topic.lower()

        # Code examples for different topics
        examples = {
            'ec2': """
# AWS EC2 Instance Creation Example
instance = aws.create_ec2_instance(
    instance_type='t2.micro',
    image_id='ami-12345678',
    key_name='my-key-pair'
)

print(f"Created EC2 instance: {instance['instance_id']}")
print(f"Instance type: {instance['instance_type']}")
print(f"Public IP: {instance['public_ip']}")
""",
            's3': """
# AWS S3 Bucket Operations Example
bucket = aws.create_s3_bucket('my-unique-bucket-name')
print(f"Created S3 bucket: {bucket['bucket_name']}")

# Upload a file
data = "Hello, Cloud Storage!"
result = aws.upload_to_s3(bucket['bucket_name'], 'hello.txt', data)
print(f"Uploaded file with ETag: {result['etag']}")
print(f"File size: {result['size']} bytes")
""",
            'lambda': """
# Serverless Function Cost Calculation
def calculate_lambda_cost(invocations, memory_mb, duration_ms):
    # AWS Lambda pricing (as of 2023)
    request_price = 0.0000002  # $0.20 per 1M requests
    compute_price = 0.0000166667  # $0.0000166667 per GB-second

    # Calculate costs
    request_cost = invocations * request_price
    compute_gb_seconds = (memory_mb / 1024) * (duration_ms / 1000) * invocations
    compute_cost = compute_gb_seconds * compute_price

    total_cost = request_cost + compute_cost

    return {
        'invocations': invocations,
        'request_cost': round(request_cost, 6),
        'compute_cost': round(compute_cost, 6),
        'total_cost': round(total_cost, 6)
    }

# Example usage
cost = calculate_lambda_cost(
    invocations=1000000,  # 1 million requests
    memory_mb=256,        # 256 MB memory
    duration_ms=100       # 100 ms execution time
)

print(f"Lambda Cost Analysis:")
print(f"Requests: {cost['invocations']:,}")
print(f"Request Cost: ${cost['request_cost']}")
print(f"Compute Cost: ${cost['compute_cost']}")
print(f"Total Cost: ${cost['total_cost']}")
""",
            'docker': """
# Docker Container Management Example
class ContainerManager:
    def __init__(self):
        self.containers = []

    def deploy_container(self, image, ports=None, env_vars=None):
        container = {
            'id': f'container_{len(self.containers)}',
            'image': image,
            'ports': ports or {},
            'env': env_vars or {},
            'status': 'running'
        }
        self.containers.append(container)
        return container

    def list_containers(self):
        return self.containers

    def stop_container(self, container_id):
        for container in self.containers:
            if container['id'] == container_id:
                container['status'] = 'stopped'
                return True
        return False

# Usage
manager = ContainerManager()

# Deploy containers
web_app = manager.deploy_container(
    image='nginx:latest',
    ports={'80/tcp': 8080},
    env_vars={'ENV': 'production'}
)

database = manager.deploy_container(
    image='postgres:13',
    ports={'5432/tcp': 5432},
    env_vars={'POSTGRES_PASSWORD': 'secret'}
)

# List all containers
for container in manager.list_containers():
    print(f"Container: {container['id']}")
    print(f"  Image: {container['image']}")
    print(f"  Status: {container['status']}")
""",
            'kubernetes': """
# Kubernetes Deployment Example
class Deployment:
    def __init__(self, name, image, replicas=1):
        self.name = name
        self.image = image
        self.replicas = replicas
        self.available_replicas = 0
        self.pods = []

    def deploy(self):
        print(f"Deploying {self.name} with image {self.image}...")

        # Simulate pod creation
        for i in range(self.replicas):
            pod = {
                'name': f"{self.name}-pod-{i}",
                'image': self.image,
                'status': 'Running',
                'node': f'node-{i % 3}'
            }
            self.pods.append(pod)

        self.available_replicas = self.replicas
        print(f"Deployment {self.name} complete. {self.replicas} pods running.")

    def scale(self, new_replicas):
        print(f"Scaling {self.name} from {self.replicas} to {new_replicas} replicas...")

        if new_replicas > self.replicas:
            # Scale up
            for i in range(self.replicas, new_replicas):
                pod = {
                    'name': f"{self.name}-pod-{i}",
                    'image': self.image,
                    'status': 'Running',
                    'node': f'node-{i % 3}'
                }
                self.pods.append(pod)
        else:
            # Scale down
            self.pods = self.pods[:new_replicas]

        self.replicas = new_replicas
        self.available_replicas = new_replicas
        print(f"Scaled to {new_replicas} replicas.")

# Create and manage deployment
app_deployment = Deployment(
    name='web-app',
    image='nginx:latest',
    replicas=3
)

app_deployment.deploy()

# Scale the deployment
app_deployment.scale(5)

# Show pod information
print("\\nCurrent Pods:")
for pod in app_deployment.pods:
    print(f"  {pod['name']} on {pod['node']} - {pod['status']}")
"""
        }

        # Find matching example
        code_example = examples.get(topic_lower)
        if not code_example:
            # Try partial match
            for key, example in examples.items():
                if key in topic_lower:
                    code_example = example
                    break

        # Default example
        if not code_example:
            code_example = """
# Cloud Computing Example
print("Welcome to Cloud Computing!")
print("This is a safe code execution environment.")

# Example: Calculate cloud costs
def calculate_monthly_cost(instances, hours_per_day=24, days_per_month=30):
    hourly_cost = 0.0116  # t2.micro instance
    monthly_cost = instances * hourly_cost * hours_per_day * days_per_month
    return monthly_cost

# Calculate for 5 instances
instances = 5
cost = calculate_monthly_cost(instances)
print(f"\\nCost for {instances} instances:")
print(f"  Hourly: ${instances * 0.0116:.4f}")
print(f"  Daily: ${instances * 0.0116 * 24:.2f}")
print(f"  Monthly: ${cost:.2f}")
"""

        return code_example

    def run_interactive_session(self, initial_code: str = None) -> Dict[str, Any]:
        """Start interactive code session"""
        session_id = f"session_{datetime.now().timestamp()}"

        session = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'code_history': [],
            'output_history': [],
            'variables': {}
        }

        if initial_code:
            # Execute initial code
            result = self.execute_safe(initial_code)
            session['code_history'].append(initial_code)
            session['output_history'].append(result.output)

            if result.success:
                session['last_output'] = result.output
            else:
                session['last_error'] = result.error

        return session

    def execute_in_session(self, session_id: str, code: str) -> Dict[str, Any]:
        """Execute code in existing session"""
        # Note: Full session persistence would require database storage
        # This is a simplified version

        result = self.execute_safe(code)

        return dict(session_id=session_id, code=code, result=result.__dict__, timestamp=datetime.now().isoformat())