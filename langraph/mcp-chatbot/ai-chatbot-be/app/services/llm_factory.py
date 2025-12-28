"""
Production LLM Factory
======================
Thread-safe singleton factory for AWS Bedrock Claude LLM.

Environment Variables Required:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION (default: us-east-1)
- BEDROCK_MODEL_ID (default: us.anthropic.claude-3-5-sonnet-20241022-v2:0)
"""

import os
import threading
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from langchain_aws import ChatBedrock

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMFactoryError(Exception):
    """LLM Factory error."""
    pass


class LLMFactory:
    """
    Thread-safe singleton factory for production LLM instances.

    Features:
    - Singleton pattern with double-checked locking
    - LLM instance caching for performance
    - AWS credential management
    - Cross-region Bedrock model support
    """

    _instance: Optional['LLMFactory'] = None
    _lock = threading.Lock()
    _cache_lock = threading.Lock()

    # Configuration defaults
    DEFAULT_REGION = "us-east-1"
    DEFAULT_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 4096

    def __new__(cls) -> 'LLMFactory':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize factory state."""
        self._llm_cache: dict = {}
        self._credentials_configured = False
        logger.info("LLMFactory singleton initialized")

    def _configure_aws_credentials(self) -> None:
        """Configure AWS credentials from environment/settings."""
        if self._credentials_configured:
            return

        # Get credentials from settings or environment
        access_key = getattr(settings, 'aws_access_key_id', None) or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = getattr(settings, 'aws_secret_access_key', None) or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = getattr(settings, 'aws_region', None) or os.getenv("AWS_REGION", self.DEFAULT_REGION)

        # Set environment variables for boto3
        if access_key:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
        if secret_key:
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)
        os.environ.setdefault("AWS_DEFAULT_REGION", region)
        os.environ.setdefault("AWS_REGION", region)

        # Configure default boto3 session
        boto3.setup_default_session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region
        )

        self._credentials_configured = True
        logger.info(f"AWS credentials configured for region: {region}")

    def create_llm(
        self,
        model_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        streaming: bool = True
    ) -> ChatBedrock:
        """
        Create or retrieve cached Bedrock LLM instance.

        Args:
            model_id: Bedrock model ID (uses BEDROCK_MODEL_ID env var if not provided)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            streaming: Enable streaming responses

        Returns:
            Configured ChatBedrock instance

        Raises:
            LLMFactoryError: If Bedrock connection fails
        """
        # Get model ID from settings, env, or default
        model_id = (
            model_id or
            getattr(settings, 'bedrock_model_id', None) or
            os.getenv("BEDROCK_MODEL_ID", self.DEFAULT_MODEL)
        )
        cache_key = f"{model_id}_{temperature}_{max_tokens}_{streaming}"

        # Return cached instance if available
        with self._cache_lock:
            if cache_key in self._llm_cache:
                logger.debug(f"Returning cached LLM: {model_id}")
                return self._llm_cache[cache_key]

        # Configure credentials
        self._configure_aws_credentials()

        try:
            # Get region
            region = getattr(settings, 'aws_region', None) or os.getenv("AWS_REGION", self.DEFAULT_REGION)

            # Create Bedrock runtime client
            client = boto3.client("bedrock-runtime", region_name=region)

            # Validate model availability (non-blocking for cross-region models)
            self._validate_model(model_id)

            # Create ChatBedrock instance
            llm = ChatBedrock(
                model_id=model_id,
                client=client,
                streaming=streaming,
                model_kwargs={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            # Cache the instance
            with self._cache_lock:
                self._llm_cache[cache_key] = llm

            logger.info(f"Bedrock LLM created: {model_id}")
            return llm

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            logger.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            raise LLMFactoryError(
                "AWS credentials missing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )
        except ClientError as e:
            logger.error(f"Bedrock client error: {e}")
            raise LLMFactoryError(f"Bedrock connection failed: {e}")
        except Exception as e:
            logger.error(f"LLM creation failed: {e}")
            raise LLMFactoryError(f"Failed to create LLM: {e}")

    def _validate_model(self, model_id: str) -> None:
        """
        Validate model is accessible.

        Cross-region models (us.anthropic.*, eu.anthropic.*, etc.) skip strict validation
        since they may not appear in the regional model list.
        """
        # Cross-region models skip validation
        if any(prefix in model_id for prefix in ["us.", "eu.", "ap."]):
            logger.info(f"Cross-region model: {model_id} (validation skipped)")
            return

        try:
            region = getattr(settings, 'aws_region', None) or os.getenv("AWS_REGION", self.DEFAULT_REGION)
            client = boto3.client("bedrock", region_name=region)
            response = client.list_foundation_models(byProvider="anthropic")
            available = [m["modelId"] for m in response.get("modelSummaries", [])]

            if model_id not in available:
                logger.warning(
                    f"Model {model_id} not in available list. "
                    f"Ensure it's enabled in AWS Console → Bedrock → Model access"
                )

        except ClientError as e:
            if "AccessDenied" in str(e):
                logger.debug("Cannot list models (AccessDenied) - assuming model available")
            else:
                raise

    def get_provider_info(self) -> str:
        """Get current LLM provider information."""
        model_id = (
            getattr(settings, 'bedrock_model_id', None) or
            os.getenv("BEDROCK_MODEL_ID", self.DEFAULT_MODEL)
        )
        return f"AWS Bedrock ({model_id})"

    def get_current_provider(self) -> str:
        """Alias for get_provider_info() for backward compatibility."""
        return self.get_provider_info()

    def clear_cache(self) -> None:
        """Clear LLM instance cache."""
        with self._cache_lock:
            self._llm_cache.clear()
        logger.info("LLM cache cleared")


# Global singleton instance
llm_factory = LLMFactory()
