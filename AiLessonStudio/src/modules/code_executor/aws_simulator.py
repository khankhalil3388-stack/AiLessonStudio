import json
import yaml
import re
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import time
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import sqlite3
from contextlib import contextmanager
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSService(Enum):
    """AWS Services supported by the simulator"""
    EC2 = "ec2"
    S3 = "s3"
    RDS = "rds"
    LAMBDA = "lambda"
    VPC = "vpc"
    IAM = "iam"
    CLOUDFRONT = "cloudfront"
    ROUTE53 = "route53"
    SQS = "sqs"
    SNS = "sns"
    DYNAMODB = "dynamodb"
    ECS = "ecs"
    EKS = "eks"
    CLOUDWATCH = "cloudwatch"


@dataclass
class EC2Instance:
    """Represents an EC2 instance"""
    instance_id: str
    instance_type: str
    state: str  # pending, running, stopped, terminated
    launch_time: datetime
    image_id: str
    key_name: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'InstanceId': self.instance_id,
            'InstanceType': self.instance_type,
            'State': {'Name': self.state},
            'LaunchTime': self.launch_time.isoformat(),
            'ImageId': self.image_id,
            'KeyName': self.key_name,
            'SecurityGroups': [{'GroupName': sg} for sg in self.security_groups],
            'Tags': [{'Key': k, 'Value': v} for k, v in self.tags.items()],
            'PublicIpAddress': self.public_ip,
            'PrivateIpAddress': self.private_ip
        }


@dataclass
class S3Bucket:
    """Represents an S3 bucket"""
    name: str
    creation_date: datetime
    region: str
    objects: Dict[str, 'S3Object'] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'Name': self.name,
            'CreationDate': self.creation_date.isoformat(),
            'Region': self.region
        }


@dataclass
class S3Object:
    """Represents an S3 object"""
    key: str
    last_modified: datetime
    size: int
    storage_class: str = 'STANDARD'
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class LambdaFunction:
    """Represents a Lambda function"""
    function_name: str
    runtime: str
    handler: str
    role: str
    code_size: int
    last_modified: datetime
    timeout: int = 3
    memory_size: int = 128
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def invoke(self, event: Dict) -> Dict:
        """Simulate Lambda invocation"""
        return {
            'StatusCode': 200,
            'FunctionError': None,
            'Payload': json.dumps({
                'statusCode': 200,
                'body': json.dumps({
                    'message': f"Function {self.function_name} executed successfully",
                    'input': event,
                    'timestamp': datetime.now().isoformat()
                })
            })
        }


class AWSSimulator:
    """Main AWS Simulator Class"""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.ec2_instances: Dict[str, EC2Instance] = {}
        self.s3_buckets: Dict[str, S3Bucket] = {}
        self.lambda_functions: Dict[str, LambdaFunction] = {}
        self.rds_instances: Dict[str, Dict] = {}
        self.vpcs: Dict[str, Dict] = {}
        self.iam_users: Dict[str, Dict] = {}
        self.iam_roles: Dict[str, Dict] = {}
        self.cloudwatch_logs: List[Dict] = []

        # Initialize SQLite database for persistent simulation
        self.db_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self._init_database()

        # Initialize with sample data
        self._initialize_sample_data()

    def _init_database(self):
        """Initialize SQLite database tables"""
        cursor = self.db_connection.cursor()

        # Create EC2 instances table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ec2_instances (
                instance_id TEXT PRIMARY KEY,
                instance_type TEXT,
                state TEXT,
                launch_time TEXT,
                image_id TEXT,
                key_name TEXT,
                public_ip TEXT,
                private_ip TEXT,
                tags TEXT
            )
        ''')

        # Create S3 buckets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS s3_buckets (
                name TEXT PRIMARY KEY,
                creation_date TEXT,
                region TEXT,
                tags TEXT
            )
        ''')

        # Create Lambda functions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lambda_functions (
                function_name TEXT PRIMARY KEY,
                runtime TEXT,
                handler TEXT,
                role TEXT,
                code_size INTEGER,
                last_modified TEXT,
                timeout INTEGER,
                memory_size INTEGER,
                environment_variables TEXT
            )
        ''')

        # Create CloudWatch logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cloudwatch_logs (
                log_id TEXT PRIMARY KEY,
                log_group TEXT,
                log_stream TEXT,
                timestamp TEXT,
                message TEXT,
                resource_type TEXT,
                resource_id TEXT
            )
        ''')

        self.db_connection.commit()

    def _log_cloudwatch(self, log_group: str, message: str,
                        resource_type: str = None, resource_id: str = None):
        """Add log entry to CloudWatch simulation"""
        log_entry = {
            'logId': str(uuid.uuid4()),
            'logGroup': log_group,
            'logStream': 'simulated-stream',
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'resourceType': resource_type,
            'resourceId': resource_id
        }

        self.cloudwatch_logs.append(log_entry)

        # Also store in database
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO cloudwatch_logs (log_id, log_group, log_stream, 
                                        timestamp, message, resource_type, resource_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_entry['logId'], log_entry['logGroup'], log_entry['logStream'],
            log_entry['timestamp'], log_entry['message'],
            log_entry['resourceType'], log_entry['resourceId']
        ))
        self.db_connection.commit()

        return log_entry

    def _initialize_sample_data(self):
        """Initialize with sample AWS resources"""
        # Create sample EC2 instances
        sample_instances = [
            EC2Instance(
                instance_id="i-0abcdef1234567890",
                instance_type="t2.micro",
                state="running",
                launch_time=datetime.now() - timedelta(days=7),
                image_id="ami-0abcdef1234567890",
                key_name="my-key-pair",
                security_groups=["default-sg"],
                tags={"Name": "WebServer-1", "Environment": "Development"},
                public_ip="54.123.45.67",
                private_ip="10.0.1.15"
            ),
            EC2Instance(
                instance_id="i-1bcdef234567890a",
                instance_type="t3.small",
                state="stopped",
                launch_time=datetime.now() - timedelta(days=3),
                image_id="ami-1bcdef234567890a",
                key_name="prod-key-pair",
                security_groups=["web-sg", "ssh-sg"],
                tags={"Name": "AppServer-1", "Environment": "Production"}
            )
        ]

        for instance in sample_instances:
            self.ec2_instances[instance.instance_id] = instance

        # Create sample S3 buckets
        sample_buckets = [
            S3Bucket(
                name="my-webapp-assets-123",
                creation_date=datetime.now() - timedelta(days=30),
                region="us-east-1",
                tags={"Environment": "Production", "Owner": "DevOps"}
            ),
            S3Bucket(
                name="logs-backup-2024",
                creation_date=datetime.now() - timedelta(days=15),
                region="us-east-1",
                tags={"Backup": "true", "Retention": "90days"}
            ),
            S3Bucket(
                name="my-web-assets-bucket",
                creation_date=datetime.now() - timedelta(days=30),
                region="us-east-1",
                tags={"Purpose": "Website", "Environment": "Production"}
            ),
            S3Bucket(
                name="backup-data-2024",
                creation_date=datetime.now() - timedelta(days=15),
                region="us-east-1",
                tags={"Purpose": "Backup", "Retention": "90-days"}
            )
        ]

        for bucket in sample_buckets:
            self.s3_buckets[bucket.name] = bucket

            # Add sample objects for the original buckets only
            if bucket.name in ["my-web-assets-bucket", "backup-data-2024"]:
                bucket.objects["index.html"] = S3Object(
                    key="index.html",
                    last_modified=datetime.now() - timedelta(days=1),
                    size=1024,
                    metadata={"Content-Type": "text/html"}
                )
                bucket.objects["styles.css"] = S3Object(
                    key="styles.css",
                    last_modified=datetime.now() - timedelta(days=1),
                    size=2048,
                    metadata={"Content-Type": "text/css"}
                )

        # Create sample Lambda functions
        sample_lambdas = [
            LambdaFunction(
                function_name="process-images",
                runtime="python3.9",
                handler="lambda_function.handler",
                role="arn:aws:iam::123456789012:role/lambda-execution-role",
                code_size=2048,
                last_modified=datetime.now() - timedelta(hours=2),
                timeout=30,
                memory_size=512,
                environment_variables={"ENVIRONMENT": "prod", "LOG_LEVEL": "INFO"}
            ),
            LambdaFunction(
                function_name="data-processor",
                runtime="nodejs18.x",
                handler="index.handler",
                role="arn:aws:iam::123456789012:role/data-processor-role",
                code_size=4096,
                last_modified=datetime.now() - timedelta(days=1),
                timeout=60,
                memory_size=1024
            ),
            LambdaFunction(
                function_name="hello-world",
                runtime="python3.9",
                handler="lambda_function.lambda_handler",
                role="arn:aws:iam::123456789012:role/lambda-role",
                code_size=512,
                last_modified=datetime.now() - timedelta(days=2)
            ),
            LambdaFunction(
                function_name="process-data",
                runtime="nodejs14.x",
                handler="index.handler",
                role="arn:aws:iam::123456789012:role/data-processor-role",
                code_size=1024,
                last_modified=datetime.now() - timedelta(days=1),
                timeout=30,
                memory_size=256,
                environment_variables={"ENVIRONMENT": "dev", "LOG_LEVEL": "info"}
            )
        ]

        for func in sample_lambdas:
            self.lambda_functions[func.function_name] = func

        logger.info(f"✅ AWS Simulator initialized in region {self.region}")

    # ============= EC2 Simulation Methods =============

    def describe_instances(self, filters: List[Dict] = None, **kwargs) -> Dict:
        """Describe EC2 instances - Combined implementation"""
        # Handle both parameter styles
        if filters is None:
            filters = kwargs.get('Filters', [])
        instance_ids = kwargs.get('InstanceIds', [])

        filtered_instances = []

        for instance in self.ec2_instances.values():
            include_instance = True

            # Apply filter matching from FIX 1
            if filters and not self._matches_filters(instance.to_dict(), filters):
                include_instance = False

            # Apply instance ID filter from existing code
            if instance_ids and instance.instance_id not in instance_ids:
                include_instance = False

            if include_instance:
                filtered_instances.append(instance)

        return {
            'Reservations': [{
                'ReservationId': 'r-' + str(uuid.uuid4())[:8],
                'Instances': [inst.to_dict() for inst in filtered_instances]
            }]
        }

    def run_instances(self, image_id: str = None, instance_type: str = "t2.micro",
                      min_count: int = 1, max_count: int = 1,
                      key_name: str = None, security_groups: List[str] = None, **kwargs) -> Dict:
        """Launch EC2 instances - Combined implementation"""
        # Handle both parameter styles
        if image_id is None:
            image_id = kwargs.get('ImageId', 'ami-0abcdef1234567890')
        if 'InstanceType' in kwargs:
            instance_type = kwargs['InstanceType']
        if 'KeyName' in kwargs:
            key_name = kwargs['KeyName']
        if 'SecurityGroups' in kwargs:
            security_groups = kwargs['SecurityGroups']

        instances_created = []

        for i in range(min_count):
            instance_id = f"i-{uuid.uuid4().hex[:17]}"
            instance = EC2Instance(
                instance_id=instance_id,
                instance_type=instance_type,
                state="running",
                launch_time=datetime.now(),
                image_id=image_id,
                key_name=key_name,
                security_groups=security_groups or ["default"],
                tags=kwargs.get('Tags', {}),
                public_ip=f"54.{random.randint(100, 255)}.{random.randint(100, 255)}.{random.randint(100, 255)}",
                private_ip=f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"
            )

            self.ec2_instances[instance_id] = instance
            instances_created.append(instance.to_dict())

            self._log_cloudwatch(
                log_group="/aws/ec2",
                message=f"Launched instance {instance_id}",
                resource_type="EC2",
                resource_id=instance_id
            )

        return {
            'Instances': instances_created,
            'ReservationId': 'r-' + str(uuid.uuid4())[:8]
        }

    def stop_instances(self, instance_ids: List[str]) -> Dict:
        """Stop EC2 instances - Combined implementation"""
        results = []
        for instance_id in instance_ids:
            if instance_id in self.ec2_instances:
                instance = self.ec2_instances[instance_id]
                previous_state = instance.state
                instance.state = "stopped"
                results.append({
                    'InstanceId': instance_id,
                    'CurrentState': {'Name': 'stopped'},
                    'PreviousState': {'Name': previous_state}
                })

                self._log_cloudwatch(
                    log_group="/aws/ec2",
                    message=f"Stopped instance {instance_id}",
                    resource_type="EC2",
                    resource_id=instance_id
                )

        return {'StoppingInstances': results}

    def terminate_instances(self, instance_ids: List[str]) -> Dict:
        """Terminate EC2 instances - Combined implementation"""
        results = []
        for instance_id in instance_ids:
            if instance_id in self.ec2_instances:
                instance = self.ec2_instances[instance_id]
                previous_state = instance.state
                instance.state = "terminated"
                results.append({
                    'InstanceId': instance_id,
                    'CurrentState': {'Name': 'terminated'},
                    'PreviousState': {'Name': previous_state}
                })

                self._log_cloudwatch(
                    log_group="/aws/ec2",
                    message=f"Terminated instance {instance_id}",
                    resource_type="EC2",
                    resource_id=instance_id
                )

        return {'TerminatingInstances': results}

    # ============= S3 Simulation Methods =============

    def create_bucket(self, bucket_name: str, region: str = None, **kwargs) -> Dict:
        """Create S3 bucket - Combined implementation"""
        if bucket_name in self.s3_buckets:
            raise Exception(f"Bucket {bucket_name} already exists")

        region = region or self.region
        bucket = S3Bucket(
            name=bucket_name,
            creation_date=datetime.now(),
            region=region,
            tags=kwargs.get('tags', kwargs.get('Tags', {}))
        )

        self.s3_buckets[bucket_name] = bucket

        self._log_cloudwatch(
            log_group="/aws/s3",
            message=f"Created bucket {bucket_name}",
            resource_type="S3",
            resource_id=bucket_name
        )

        return {
            'Location': f"/{bucket_name}",
            'Bucket': bucket.to_dict()
        }

    def list_buckets(self) -> Dict:
        """List S3 buckets - Combined implementation"""
        buckets = [bucket.to_dict() for bucket in self.s3_buckets.values()]
        return {
            'Buckets': buckets,
            'Owner': {'DisplayName': 'simulated-owner'}
        }

    def put_object(self, bucket: str, key: str, body: bytes,
                   metadata: Dict = None, **kwargs) -> Dict:
        """Put object in S3 bucket - Combined implementation"""
        if bucket not in self.s3_buckets:
            raise Exception(f"Bucket {bucket} does not exist")

        # Merge metadata from both parameter styles
        merged_metadata = metadata or {}
        if 'Metadata' in kwargs:
            merged_metadata.update(kwargs['Metadata'])

        s3_object = S3Object(
            key=key,
            last_modified=datetime.now(),
            size=len(body),
            storage_class=kwargs.get('StorageClass', 'STANDARD'),
            metadata=merged_metadata
        )

        self.s3_buckets[bucket].objects[key] = s3_object

        self._log_cloudwatch(
            log_group="/aws/s3",
            message=f"Put object {key} in bucket {bucket}",
            resource_type="S3",
            resource_id=bucket
        )

        return {
            'ETag': hashlib.md5(body).hexdigest(),
            'VersionId': 'null'
        }

    def get_object(self, bucket: str, key: str) -> Dict:
        """Simulate S3 GetObject API"""
        if bucket not in self.s3_buckets:
            raise Exception(f"Bucket {bucket} does not exist")

        if key not in self.s3_buckets[bucket].objects:
            raise Exception(f"Key {key} does not exist in bucket {bucket}")

        s3_object = self.s3_buckets[bucket].objects[key]

        # Simulate file content
        content = f"Simulated content for {key}".encode()

        return {
            'Body': io.BytesIO(content),
            'LastModified': s3_object.last_modified,
            'ContentLength': s3_object.size,
            'ETag': hashlib.md5(content).hexdigest(),
            'Metadata': s3_object.metadata
        }

    def list_objects(self, bucket: str, prefix: str = "") -> Dict:
        """List objects in S3 bucket - from FIX 1"""
        if bucket not in self.s3_buckets:
            raise Exception(f"Bucket {bucket} does not exist")

        objects = []
        for key, obj in self.s3_buckets[bucket].objects.items():
            if prefix and not key.startswith(prefix):
                continue
            objects.append({
                'Key': key,
                'LastModified': obj.last_modified.isoformat(),
                'Size': obj.size,
                'StorageClass': obj.storage_class
            })

        return {
            'Contents': objects,
            'Name': bucket,
            'Prefix': prefix,
            'MaxKeys': 1000,
            'IsTruncated': False
        }

    def list_objects_v2(self, bucket: str, **kwargs) -> Dict:
        """Simulate S3 ListObjectsV2 API"""
        if bucket not in self.s3_buckets:
            raise Exception(f"Bucket {bucket} does not exist")

        prefix = kwargs.get('Prefix', '')
        objects_list = []

        for key, obj in self.s3_buckets[bucket].objects.items():
            if key.startswith(prefix):
                objects_list.append({
                    'Key': key,
                    'LastModified': obj.last_modified,
                    'Size': obj.size,
                    'StorageClass': obj.storage_class
                })

        return {
            'Contents': objects_list,
            'KeyCount': len(objects_list),
            'MaxKeys': 1000,
            'IsTruncated': False
        }

    # ============= Lambda Simulation Methods =============

    def create_function(self, function_name: str = None, runtime: str = None,
                        handler: str = None, role: str = None, code: Dict = None, **kwargs) -> Dict:
        """Create Lambda function - Combined implementation"""
        # Handle both parameter styles
        if function_name is None:
            function_name = kwargs['FunctionName']
        if runtime is None:
            runtime = kwargs['Runtime']
        if handler is None:
            handler = kwargs['Handler']
        if role is None:
            role = kwargs['Role']
        if code is None:
            code = kwargs.get('Code', {})

        if function_name in self.lambda_functions:
            raise Exception(f"Function {function_name} already exists")

        # Determine code size
        code_size = code.get('ZipFile', b'').__sizeof__()
        if 'Code' in kwargs and 'ZipFile' in kwargs['Code']:
            code_size = len(kwargs['Code']['ZipFile'])

        function = LambdaFunction(
            function_name=function_name,
            runtime=runtime,
            handler=handler,
            role=role,
            code_size=code_size,
            last_modified=datetime.now(),
            timeout=kwargs.get('Timeout', code.get('Timeout', 3)),
            memory_size=kwargs.get('MemorySize', code.get('MemorySize', 128)),
            environment_variables=kwargs.get('Environment', {}).get('Variables',
                                                                    code.get('Environment', {}).get('Variables', {}))
        )

        self.lambda_functions[function_name] = function

        self._log_cloudwatch(
            log_group="/aws/lambda",
            message=f"Created function {function_name}",
            resource_type="Lambda",
            resource_id=function_name
        )

        return {
            'FunctionName': function.function_name,
            'FunctionArn': f"arn:aws:lambda:{self.region}:123456789012:function:{function_name}",
            'Runtime': function.runtime,
            'Handler': function.handler,
            'Role': function.role,
            'LastModified': function.last_modified.isoformat(),
            'CodeSize': function.code_size,
            'Description': kwargs.get('Description', ''),
            'Timeout': function.timeout,
            'MemorySize': function.memory_size
        }

    def invoke(self, function_name: str, payload: bytes = None) -> Dict:
        """Invoke Lambda function - Combined implementation"""
        if function_name not in self.lambda_functions:
            raise Exception(f"Function {function_name} not found")

        function = self.lambda_functions[function_name]
        event = json.loads(payload.decode('utf-8')) if payload else {}

        self._log_cloudwatch(
            log_group=f"/aws/lambda/{function_name}",
            message=f"Invoked function {function_name}",
            resource_type="Lambda",
            resource_id=function_name
        )

        return function.invoke(event)

    def list_functions(self) -> Dict:
        """List Lambda functions - Combined implementation"""
        functions = []
        for func in self.lambda_functions.values():
            functions.append({
                'FunctionName': func.function_name,
                'Runtime': func.runtime,
                'Handler': func.handler,
                'Role': func.role,
                'CodeSize': func.code_size,
                'LastModified': func.last_modified.isoformat(),
                'Timeout': func.timeout,
                'MemorySize': func.memory_size,
                'Environment': {'Variables': func.environment_variables},
                'FunctionArn': f"arn:aws:lambda:{self.region}:123456789012:function:{func.function_name}",
                'Description': ''
            })

        return {'Functions': functions}

    # ============= IAM Simulation Methods =============

    def create_user(self, user_name: str) -> Dict:
        """Simulate IAM CreateUser API"""
        user_id = f"AID{hashlib.md5(user_name.encode()).hexdigest()[:16].upper()}"

        user = {
            'UserName': user_name,
            'UserId': user_id,
            'Arn': f"arn:aws:iam::123456789012:user/{user_name}",
            'CreateDate': datetime.now(),
            'Policies': [],
            'Groups': []
        }

        self.iam_users[user_name] = user

        self._log_cloudwatch(
            log_group="/aws/iam",
            message=f"Created IAM user {user_name}",
            resource_type="IAM",
            resource_id=user_name
        )

        return {'User': user}

    def attach_user_policy(self, user_name: str, policy_arn: str) -> Dict:
        """Simulate IAM AttachUserPolicy API"""
        if user_name not in self.iam_users:
            raise Exception(f"User {user_name} not found")

        self.iam_users[user_name]['Policies'].append(policy_arn)

        return {}

    def create_role(self, role_name: str, assume_role_policy_document: Dict) -> Dict:
        """Simulate IAM CreateRole API"""
        role_id = f"ARO{hashlib.md5(role_name.encode()).hexdigest()[:16].upper()}"

        role = {
            'RoleName': role_name,
            'RoleId': role_id,
            'Arn': f"arn:aws:iam::123456789012:role/{role_name}",
            'CreateDate': datetime.now(),
            'AssumeRolePolicyDocument': assume_role_policy_document,
            'Description': '',
            'MaxSessionDuration': 3600,
            'Path': '/',
            'Policies': []
        }

        self.iam_roles[role_name] = role

        self._log_cloudwatch(
            log_group="/aws/iam",
            message=f"Created IAM role {role_name}",
            resource_type="IAM",
            resource_id=role_name
        )

        return {'Role': role}

    # ============= CloudWatch Simulation Methods =============

    def get_log_events(self, log_group_name: str, limit: int = 10, **kwargs) -> Dict:
        """Get CloudWatch log events - Combined implementation"""
        events = []

        # Use limit from either parameter
        actual_limit = kwargs.get('limit', limit)

        for log in reversed(self.cloudwatch_logs):
            if log['logGroup'] == log_group_name:
                events.append({
                    'timestamp': int(datetime.fromisoformat(log['timestamp']).timestamp() * 1000),
                    'message': log['message'],
                    'ingestionTime': int(time.time() * 1000),
                    'logStreamName': log['logStream']
                })
                if len(events) >= actual_limit:
                    break

        return {'events': events}

    # ============= Helper Methods =============

    def _matches_filters(self, instance_data: Dict, filters: List[Dict]) -> bool:
        """Check if instance matches filters"""
        for filter_item in filters:
            for key, values in filter_item.items():
                instance_value = self._get_nested_value(instance_data, key)
                if not any(str(instance_value) == str(v) for v in values):
                    return False
        return True

    def _get_nested_value(self, data: Dict, key: str):
        """Get nested value from dictionary"""
        keys = key.split('.')
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
            elif isinstance(data, list) and k.isdigit():
                data = data[int(k)] if int(k) < len(data) else None
            else:
                return None
        return data

    def get_resource_counts(self) -> Dict:
        """Get counts of all resources"""
        return {
            'ec2_instances': len(self.ec2_instances),
            's3_buckets': len(self.s3_buckets),
            'lambda_functions': len(self.lambda_functions),
            'cloudwatch_logs': len(self.cloudwatch_logs),
            'iam_users': len(self.iam_users),
            'iam_roles': len(self.iam_roles)
        }

    def get_service_status(self) -> Dict:
        """Get status of all simulated services"""
        return {
            'EC2': {
                'total_instances': len(self.ec2_instances),
                'running': len([i for i in self.ec2_instances.values() if i.state == 'running']),
                'stopped': len([i for i in self.ec2_instances.values() if i.state == 'stopped']),
                'terminated': len([i for i in self.ec2_instances.values() if i.state == 'terminated'])
            },
            'S3': {
                'total_buckets': len(self.s3_buckets),
                'total_objects': sum(len(b.objects) for b in self.s3_buckets.values())
            },
            'Lambda': {
                'total_functions': len(self.lambda_functions)
            },
            'IAM': {
                'total_users': len(self.iam_users),
                'total_roles': len(self.iam_roles)
            }
        }

    def cleanup_old_resources(self, days_old: int = 7):
        """Clean up old resources for simulation"""
        cutoff = datetime.now() - timedelta(days=days_old)

        # Clean old logs
        self.cloudwatch_logs = [
            log for log in self.cloudwatch_logs
            if datetime.fromisoformat(log['timestamp']) > cutoff
        ]

        logger.info(f"Cleaned up resources older than {days_old} days")

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.ec2_instances.clear()
        self.s3_buckets.clear()
        self.lambda_functions.clear()
        self.rds_instances.clear()
        self.vpcs.clear()
        self.iam_users.clear()
        self.iam_roles.clear()
        self.cloudwatch_logs.clear()

        # Reinitialize sample data
        self._initialize_sample_data()

        # Reset database
        cursor = self.db_connection.cursor()
        cursor.execute("DELETE FROM ec2_instances")
        cursor.execute("DELETE FROM s3_buckets")
        cursor.execute("DELETE FROM lambda_functions")
        cursor.execute("DELETE FROM cloudwatch_logs")
        self.db_connection.commit()

    def export_state(self) -> str:
        """Export simulator state as JSON"""
        state = {
            'region': self.region,
            'ec2_instances': {k: asdict(v) for k, v in self.ec2_instances.items()},
            's3_buckets': {k: asdict(v) for k, v in self.s3_buckets.items()},
            'lambda_functions': {k: asdict(v) for k, v in self.lambda_functions.items()},
            'cloudwatch_logs': self.cloudwatch_logs,
            'iam_users': self.iam_users,
            'iam_roles': self.iam_roles,
            'export_time': datetime.now().isoformat()
        }
        return json.dumps(state, indent=2, default=str)

    def export_configuration(self, format: str = 'json') -> str:
        """Export current simulation configuration"""
        config = {
            'region': self.region,
            'timestamp': datetime.now().isoformat(),
            'services': self.get_service_status(),
            'resources': {
                'ec2_instances': [inst.to_dict() for inst in self.ec2_instances.values()],
                's3_buckets': [bucket.to_dict() for bucket in self.s3_buckets.values()],
                'lambda_functions': [
                    {
                        'function_name': func.function_name,
                        'runtime': func.runtime,
                        'handler': func.handler
                    }
                    for func in self.lambda_functions.values()
                ]
            }
        }

        if format == 'json':
            return json.dumps(config, indent=2, default=str)
        elif format == 'yaml':
            return yaml.dump(config, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_state(self, state_json: str):
        """Import simulator state from JSON"""
        state = json.loads(state_json)

        self.region = state.get('region', self.region)

        # Clear existing state
        self.ec2_instances.clear()
        self.s3_buckets.clear()
        self.lambda_functions.clear()

        # Import EC2 instances
        for instance_id, instance_data in state.get('ec2_instances', {}).items():
            instance = EC2Instance(**instance_data)
            instance.launch_time = datetime.fromisoformat(instance_data['launch_time'])
            self.ec2_instances[instance_id] = instance

        # Import S3 buckets
        for bucket_name, bucket_data in state.get('s3_buckets', {}).items():
            bucket = S3Bucket(**bucket_data)
            bucket.creation_date = datetime.fromisoformat(bucket_data['creation_date'])
            self.s3_buckets[bucket_name] = bucket

        # Import Lambda functions
        for func_name, func_data in state.get('lambda_functions', {}).items():
            function = LambdaFunction(**func_data)
            function.last_modified = datetime.fromisoformat(func_data['last_modified'])
            self.lambda_functions[func_name] = function

        self.cloudwatch_logs = state.get('cloudwatch_logs', [])
        self.iam_users = state.get('iam_users', {})
        self.iam_roles = state.get('iam_roles', {})

        logger.info("Imported simulator state")

    def simulate_cost(self) -> Dict:
        """Simulate AWS cost calculation"""
        total_cost = 0
        breakdown = {}

        # EC2 costs
        ec2_cost = 0
        for instance in self.ec2_instances.values():
            if instance.state == 'running':
                # Approximate hourly cost based on instance type
                hourly_rates = {
                    't2.micro': 0.0116,
                    't2.small': 0.023,
                    't3.micro': 0.0104,
                    't3.small': 0.0208,
                    'm5.large': 0.096
                }
                rate = hourly_rates.get(instance.instance_type, 0.05)
                ec2_cost += rate * 24 * 30  # Monthly cost

        if ec2_cost > 0:
            breakdown['EC2'] = {'monthly': round(ec2_cost, 2), 'unit': 'USD'}
            total_cost += ec2_cost

        # S3 costs (simplified)
        s3_objects = sum(len(b.objects) for b in self.s3_buckets.values())
        s3_cost = s3_objects * 0.023  # Simplified cost per object
        if s3_cost > 0:
            breakdown['S3'] = {'monthly': round(s3_cost, 2), 'unit': 'USD'}
            total_cost += s3_cost

        # Lambda costs (simplified)
        lambda_cost = len(self.lambda_functions) * 0.20
        if lambda_cost > 0:
            breakdown['Lambda'] = {'monthly': round(lambda_cost, 2), 'unit': 'USD'}
            total_cost += lambda_cost

        return {
            'total_monthly_cost': round(total_cost, 2),
            'currency': 'USD',
            'region': self.region,
            'breakdown': breakdown,
            'estimation_date': datetime.now().isoformat()
        }

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()


# CLI interface for testing
if __name__ == "__main__":
    print("🔧 Testing AWS Simulator...")
    simulator = AWSSimulator()

    # Test EC2 operations
    print("\n1. Listing EC2 instances:")
    instances = simulator.describe_instances()
    print(f"   Found {len(instances['Reservations'][0]['Instances'])} instances")

    # Test S3 operations
    print("\n2. Listing S3 buckets:")
    buckets = simulator.list_buckets()
    print(f"   Found {len(buckets['Buckets'])} buckets")

    # Test Lambda operations
    print("\n3. Listing Lambda functions:")
    functions = simulator.list_functions()
    print(f"   Found {len(functions['Functions'])} functions")

    print("\n4. Getting resource counts:")
    counts = simulator.get_resource_counts()
    print(f"   EC2: {counts['ec2_instances']}, S3: {counts['s3_buckets']}, Lambda: {counts['lambda_functions']}")

    print("\n5. Testing cost simulation:")
    costs = simulator.simulate_cost()
    print(f"   Total monthly cost: ${costs['total_monthly_cost']}")

    print("\n✅ AWS Simulator working correctly!")