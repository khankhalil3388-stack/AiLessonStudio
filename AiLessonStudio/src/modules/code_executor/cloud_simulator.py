import json
import random
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime, timedelta
import threading
from queue import Queue
import asyncio


class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    GENERIC = "generic"


class ResourceType(Enum):
    VM = "virtual_machine"
    CONTAINER = "container"
    FUNCTION = "function"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    LOAD_BALANCER = "load_balancer"


@dataclass
class CloudResource:
    """Base class for cloud resources"""
    resource_id: str
    name: str
    resource_type: ResourceType
    provider: CloudProvider
    region: str
    status: str
    created_at: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'resource_id': self.resource_id,
            'name': self.name,
            'type': self.resource_type.value,
            'provider': self.provider.value,
            'region': self.region,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class VirtualMachine(CloudResource):
    """Virtual Machine resource"""
    instance_type: str
    vcpus: int
    memory_gb: int
    storage_gb: int
    os_type: str
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'instance_type': self.instance_type,
            'vcpus': self.vcpus,
            'memory_gb': self.memory_gb,
            'storage_gb': self.storage_gb,
            'os_type': self.os_type,
            'public_ip': self.public_ip,
            'private_ip': self.private_ip
        })
        return data


@dataclass
class StorageBucket(CloudResource):
    """Storage bucket resource"""
    bucket_name: str
    storage_class: str
    versioning_enabled: bool
    encryption_enabled: bool
    objects: List[Dict[str, Any]]

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'bucket_name': self.bucket_name,
            'storage_class': self.storage_class,
            'versioning_enabled': self.versioning_enabled,
            'encryption_enabled': self.encryption_enabled,
            'object_count': len(self.objects)
        })
        return data


@dataclass
class DatabaseInstance(CloudResource):
    """Database instance resource"""
    engine: str
    version: str
    storage_gb: int
    instance_class: str
    endpoint: str
    port: int

    def to_dict(self):
        data = super().to_dict()
        data.update({
            'engine': self.engine,
            'version': self.version,
            'storage_gb': self.storage_gb,
            'instance_class': self.instance_class,
            'endpoint': self.endpoint,
            'port': self.port
        })
        return data


class CloudSimulator:
    """Complete cloud environment simulator"""

    def __init__(self, config):
        self.config = config
        self.resources: Dict[str, CloudResource] = {}
        self.resource_counters = {}
        self.regions = self._initialize_regions()
        self.activity_log = []
        self.cost_tracker = CostTracker()
        self._init_resource_templates()

        # Start background tasks
        self._start_background_tasks()

        print("✅ Cloud Simulator initialized")

    def _initialize_regions(self) -> Dict[str, List[str]]:
        """Initialize available regions for each provider"""
        return {
            CloudProvider.AWS.value: [
                'us-east-1', 'us-west-2', 'eu-west-1',
                'ap-south-1', 'sa-east-1'
            ],
            CloudProvider.AZURE.value: [
                'eastus', 'westus', 'westeurope',
                'southeastasia', 'brazilsouth'
            ],
            CloudProvider.GCP.value: [
                'us-central1', 'europe-west1', 'asia-east1',
                'australia-southeast1'
            ]
        }

    def _init_resource_templates(self):
        """Initialize resource templates"""
        self.instance_templates = {
            't2.micro': {'vcpus': 1, 'memory_gb': 1, 'cost_per_hour': 0.0116},
            't2.small': {'vcpus': 1, 'memory_gb': 2, 'cost_per_hour': 0.023},
            't2.medium': {'vcpus': 2, 'memory_gb': 4, 'cost_per_hour': 0.0464},
            'm5.large': {'vcpus': 2, 'memory_gb': 8, 'cost_per_hour': 0.096},
            'c5.xlarge': {'vcpus': 4, 'memory_gb': 8, 'cost_per_hour': 0.17}
        }

        self.storage_templates = {
            'standard': {'cost_per_gb_month': 0.023},
            'infrequent': {'cost_per_gb_month': 0.0125},
            'glacier': {'cost_per_gb_month': 0.004}
        }

    def _start_background_tasks(self):
        """Start background simulation tasks"""
        # Simulate resource state changes
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_loop,
            daemon=True
        )
        self.simulation_thread.start()

    def _run_simulation_loop(self):
        """Background simulation loop"""
        while True:
            time.sleep(10)  # Run every 10 seconds

            # Update resource states
            self._simulate_resource_changes()

            # Update costs
            self.cost_tracker.update_costs(self.resources)

    def _simulate_resource_changes(self):
        """Simulate random resource state changes"""
        for resource_id, resource in list(self.resources.items()):
            # 5% chance of random state change
            if random.random() < 0.05:
                if resource.status == 'running':
                    # Simulate occasional failures
                    if random.random() < 0.01:  # 1% chance of failure
                        resource.status = 'failed'
                        self._log_activity(
                            'ERROR',
                            f'Resource {resource.name} ({resource_id}) failed unexpectedly'
                        )
                elif resource.status == 'stopped':
                    # Auto-recovery simulation
                    if random.random() < 0.1:  # 10% chance of auto-recovery
                        resource.status = 'running'
                        self._log_activity(
                            'INFO',
                            f'Resource {resource.name} ({resource_id}) auto-recovered'
                        )

    def _log_activity(self, level: str, message: str):
        """Log simulation activity"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.activity_log.append(log_entry)

        # Keep only last 1000 entries
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]

    def create_virtual_machine(self, name: str, instance_type: str = 't2.micro',
                               provider: str = 'aws', region: str = None) -> Dict[str, Any]:
        """Create a virtual machine"""
        if provider not in [p.value for p in CloudProvider]:
            provider = CloudProvider.AWS.value

        if region is None:
            region = self.regions[provider][0]

        # Validate instance type
        if instance_type not in self.instance_templates:
            instance_type = 't2.micro'

        template = self.instance_templates[instance_type]

        # Generate resource ID
        resource_id = f"{provider}-vm-{uuid.uuid4().hex[:8]}"

        # Create VM resource
        vm = VirtualMachine(
            resource_id=resource_id,
            name=name,
            resource_type=ResourceType.VM,
            provider=CloudProvider(provider),
            region=region,
            status='pending',
            created_at=datetime.now(),
            tags={'simulated': 'true', 'purpose': 'education'},
            metadata={'template': instance_type},
            instance_type=instance_type,
            vcpus=template['vcpus'],
            memory_gb=template['memory_gb'],
            storage_gb=20,  # Default storage
            os_type='linux',
            public_ip=f"{random.randint(10, 255)}.{random.randint(0, 255)}."
                      f"{random.randint(0, 255)}.{random.randint(1, 254)}",
            private_ip=f"172.{random.randint(16, 31)}."
                       f"{random.randint(0, 255)}.{random.randint(1, 254)}"
        )

        # Add to resources
        self.resources[resource_id] = vm

        # Simulate provisioning delay
        time.sleep(0.5)  # Simulated delay
        vm.status = 'running'

        # Log activity
        self._log_activity(
            'INFO',
            f'Created VM {name} ({resource_id}) with type {instance_type} in {region}'
        )

        # Track cost
        self.cost_tracker.add_resource(vm)

        return vm.to_dict()

    def create_storage_bucket(self, name: str, provider: str = 'aws',
                              region: str = None, storage_class: str = 'standard') -> Dict[str, Any]:
        """Create a storage bucket"""
        if provider not in [p.value for p in CloudProvider]:
            provider = CloudProvider.AWS.value

        if region is None:
            region = self.regions[provider][0]

        # Generate resource ID
        resource_id = f"{provider}-bucket-{uuid.uuid4().hex[:8]}"

        # Create bucket resource
        bucket = StorageBucket(
            resource_id=resource_id,
            name=name,
            resource_type=ResourceType.STORAGE,
            provider=CloudProvider(provider),
            region=region,
            status='active',
            created_at=datetime.now(),
            tags={'simulated': 'true', 'purpose': 'education'},
            metadata={'storage_class': storage_class},
            bucket_name=name,
            storage_class=storage_class,
            versioning_enabled=False,
            encryption_enabled=True,
            objects=[]
        )

        # Add to resources
        self.resources[resource_id] = bucket

        # Log activity
        self._log_activity(
            'INFO',
            f'Created storage bucket {name} ({resource_id}) in {region}'
        )

        # Track cost
        self.cost_tracker.add_resource(bucket)

        return bucket.to_dict()

    def create_database(self, name: str, engine: str = 'mysql',
                        provider: str = 'aws', region: str = None) -> Dict[str, Any]:
        """Create a database instance"""
        if provider not in [p.value for p in CloudProvider]:
            provider = CloudProvider.AWS.value

        if region is None:
            region = self.regions[provider][0]

        # Generate resource ID
        resource_id = f"{provider}-db-{uuid.uuid4().hex[:8]}"

        # Create database resource
        db = DatabaseInstance(
            resource_id=resource_id,
            name=name,
            resource_type=ResourceType.DATABASE,
            provider=CloudProvider(provider),
            region=region,
            status='available',
            created_at=datetime.now(),
            tags={'simulated': 'true', 'purpose': 'education'},
            metadata={'engine': engine},
            engine=engine,
            version='8.0' if engine == 'mysql' else '13.0',
            storage_gb=20,
            instance_class='db.t3.micro',
            endpoint=f"{resource_id}.{provider}.{region}.simulated.com",
            port=3306 if engine == 'mysql' else 5432
        )

        # Add to resources
        self.resources[resource_id] = db

        # Simulate provisioning delay
        time.sleep(0.3)

        # Log activity
        self._log_activity(
            'INFO',
            f'Created database {name} ({resource_id}) with engine {engine} in {region}'
        )

        # Track cost
        self.cost_tracker.add_resource(db)

        return db.to_dict()

    def list_resources(self, resource_type: str = None,
                       provider: str = None) -> List[Dict[str, Any]]:
        """List all resources with optional filtering"""
        filtered_resources = []

        for resource in self.resources.values():
            # Apply filters
            if resource_type and resource.resource_type.value != resource_type:
                continue
            if provider and resource.provider.value != provider:
                continue

            filtered_resources.append(resource.to_dict())

        return filtered_resources

    def get_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Get specific resource by ID"""
        resource = self.resources.get(resource_id)
        return resource.to_dict() if resource else None

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a resource"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]

            # Check if resource can be deleted
            if resource.status in ['running', 'stopped', 'available']:
                # Simulate deletion delay
                time.sleep(0.2)

                # Remove from resources
                del self.resources[resource_id]

                # Remove from cost tracker
                self.cost_tracker.remove_resource(resource_id)

                # Log activity
                self._log_activity(
                    'INFO',
                    f'Deleted resource {resource.name} ({resource_id})'
                )

                return True

        return False

    def start_resource(self, resource_id: str) -> bool:
        """Start a resource"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]

            if resource.status == 'stopped':
                resource.status = 'running'

                # Log activity
                self._log_activity(
                    'INFO',
                    f'Started resource {resource.name} ({resource_id})'
                )

                return True

        return False

    def stop_resource(self, resource_id: str) -> bool:
        """Stop a resource"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]

            if resource.status == 'running':
                resource.status = 'stopped'

                # Log activity
                self._log_activity(
                    'INFO',
                    f'Stopped resource {resource.name} ({resource_id})'
                )

                return True

        return False

    def execute_command(self, resource_id: str, command: str) -> Dict[str, Any]:
        """Execute command on resource (simulated)"""
        if resource_id not in self.resources:
            return {'success': False, 'output': 'Resource not found'}

        resource = self.resources[resource_id]

        # Simulate command execution
        time.sleep(0.1)

        # Generate simulated output based on command
        if 'ls' in command or 'dir' in command:
            output = """total 32
drwxr-xr-x  2 user user  4096 Jan 15 10:30 .
drwxr-xr-x 18 user user  4096 Jan 14 09:15 ..
-rw-r--r--  1 user user   120 Jan 15 10:30 app.py
-rw-r--r--  1 user user  1024 Jan 15 10:29 data.txt
-rw-r--r--  1 user user  2048 Jan 15 10:28 config.json
drwxr-xr-x  2 user user  4096 Jan 15 10:27 logs"""
        elif 'pwd' in command:
            output = "/home/user/cloud-project"
        elif 'whoami' in command:
            output = "user"
        elif 'df' in command or 'disk' in command:
            output = """Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        20G   5G   15G  25% /
tmpfs           2.0G     0  2.0G   0% /dev/shm"""
        else:
            output = f"Command executed: {command}\nSimulated output for educational purposes."

        # Log activity
        self._log_activity(
            'INFO',
            f'Executed command on {resource.name} ({resource_id}): {command[:50]}...'
        )

        return {
            'success': True,
            'output': output,
            'resource': resource.name,
            'command': command,
            'timestamp': datetime.now().isoformat()
        }

    def upload_file(self, bucket_id: str, file_name: str,
                    content: str = "") -> Dict[str, Any]:
        """Upload file to storage bucket"""
        if bucket_id not in self.resources:
            return {'success': False, 'message': 'Bucket not found'}

        resource = self.resources[bucket_id]
        if not isinstance(resource, StorageBucket):
            return {'success': False, 'message': 'Not a storage bucket'}

        # Create file object
        file_obj = {
            'name': file_name,
            'size': len(content),
            'upload_time': datetime.now().isoformat(),
            'content_type': self._detect_content_type(file_name),
            'etag': f"simulated-etag-{uuid.uuid4().hex[:16]}"
        }

        resource.objects.append(file_obj)

        # Log activity
        self._log_activity(
            'INFO',
            f'Uploaded file {file_name} to bucket {resource.name}'
        )

        return {
            'success': True,
            'message': f'File {file_name} uploaded successfully',
            'file': file_obj,
            'url': f"https://{resource.bucket_name}.s3.{resource.region}.amazonaws.com/{file_name}"
        }

    def _detect_content_type(self, file_name: str) -> str:
        """Detect content type from file extension"""
        extension_map = {
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.html': 'text/html',
            '.css': 'text/css',
            '.jpg': 'image/jpeg',
            '.png': 'image/png',
            '.pdf': 'application/pdf'
        }

        for ext, content_type in extension_map.items():
            if file_name.lower().endswith(ext):
                return content_type

        return 'application/octet-stream'

    def get_metrics(self, resource_id: str) -> Dict[str, Any]:
        """Get metrics for a resource"""
        if resource_id not in self.resources:
            return {}

        resource = self.resources[resource_id]

        # Generate simulated metrics
        metrics = {
            'cpu_utilization': random.uniform(5, 40),
            'memory_utilization': random.uniform(20, 70),
            'disk_utilization': random.uniform(10, 50),
            'network_in': random.uniform(100, 1000),
            'network_out': random.uniform(50, 500),
            'request_count': random.randint(100, 1000),
            'error_rate': random.uniform(0, 2),
            'timestamp': datetime.now().isoformat()
        }

        return metrics

    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost report for all resources"""
        return self.cost_tracker.get_report()

    def get_activity_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get activity log"""
        return self.activity_log[-limit:] if self.activity_log else []

    def simulate_error(self, error_type: str = 'network') -> Dict[str, Any]:
        """Simulate a cloud error for learning purposes"""
        error_scenarios = {
            'network': {
                'error': 'NetworkTimeoutError',
                'message': 'Request timed out after 30 seconds',
                'resolution': 'Check network connectivity and retry with exponential backoff',
                'prevention': 'Implement proper timeout handling and retry logic'
            },
            'permission': {
                'error': 'AccessDeniedException',
                'message': 'User is not authorized to perform this action',
                'resolution': 'Check IAM policies and permissions',
                'prevention': 'Follow principle of least privilege'
            },
            'quota': {
                'error': 'QuotaExceededException',
                'message': 'Service quota exceeded for this resource type',
                'resolution': 'Request quota increase or optimize resource usage',
                'prevention': 'Monitor usage and set up alerts'
            },
            'resource': {
                'error': 'ResourceNotFoundException',
                'message': 'The specified resource does not exist',
                'resolution': 'Verify resource ID and region',
                'prevention': 'Implement proper error handling and validation'
            }
        }

        scenario = error_scenarios.get(error_type, error_scenarios['network'])

        # Log the simulated error
        self._log_activity('ERROR', f'Simulated error: {scenario["error"]}')

        return {
            'simulated_error': True,
            'scenario': error_type,
            'details': scenario,
            'learning_objective': 'Understand common cloud errors and their resolution',
            'timestamp': datetime.now().isoformat()
        }


class CostTracker:
    """Track costs of simulated resources"""

    def __init__(self):
        self.resource_costs = {}
        self.daily_costs = {}
        self.monthly_cost = 0.0

    def add_resource(self, resource):
        """Add resource to cost tracking"""
        cost = self._calculate_hourly_cost(resource)
        self.resource_costs[resource.resource_id] = {
            'resource': resource.name,
            'type': resource.resource_type.value,
            'hourly_cost': cost,
            'added_at': datetime.now(),
            'total_cost': 0.0
        }

    def remove_resource(self, resource_id):
        """Remove resource from cost tracking"""
        if resource_id in self.resource_costs:
            del self.resource_costs[resource_id]

    def _calculate_hourly_cost(self, resource) -> float:
        """Calculate hourly cost for resource"""
        base_costs = {
            ResourceType.VM.value: 0.0116,  # t2.micro
            ResourceType.STORAGE.value: 0.023 / 730,  # per hour (monthly/730)
            ResourceType.DATABASE.value: 0.017,  # db.t3.micro
            ResourceType.CONTAINER.value: 0.008,
            ResourceType.FUNCTION.value: 0.0000002,
            ResourceType.LOAD_BALANCER.value: 0.0225,
            ResourceType.NETWORK.value: 0.01
        }

        return base_costs.get(resource.resource_type.value, 0.01)

    def update_costs(self, resources):
        """Update costs for all resources"""
        current_hour = datetime.now().hour

        for resource_id, resource in resources.items():
            if resource_id in self.resource_costs:
                cost_entry = self.resource_costs[resource_id]

                # Only charge for running resources
                if resource.status == 'running':
                    # Add hourly cost
                    cost_entry['total_cost'] += cost_entry['hourly_cost']

                    # Update monthly total
                    self.monthly_cost += cost_entry['hourly_cost']

        # Update daily costs
        today = datetime.now().date().isoformat()
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0

        # Sum all resource costs for today
        self.daily_costs[today] = sum(
            entry['total_cost'] for entry in self.resource_costs.values()
        )

    def get_report(self) -> Dict[str, Any]:
        """Generate cost report"""
        total_hourly = sum(entry['hourly_cost'] for entry in self.resource_costs.values())
        total_to_date = sum(entry['total_cost'] for entry in self.resource_costs.values())

        return {
            'total_resources': len(self.resource_costs),
            'estimated_monthly_cost': self.monthly_cost * 730,  # Extrapolate
            'current_hourly_cost': total_hourly,
            'cost_to_date': total_to_date,
            'daily_costs': self.daily_costs,
            'resource_breakdown': [
                {
                    'resource': entry['resource'],
                    'type': entry['type'],
                    'hourly_cost': entry['hourly_cost'],
                    'total_cost': entry['total_cost']
                }
                for entry in self.resource_costs.values()
            ]
        }