#!/usr/bin/env python3
"""
AI Service Startup Script
Handles environment validation, dependency checks, and graceful startup
Usage: python app/start.py [--dev] [--port PORT] [--host HOST]
"""

import os
import sys
import asyncio
import argparse
import signal
import logging
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir.parent))

try:
    from dotenv import load_dotenv
    import uvicorn
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Please ensure all dependencies are installed: pip install -r requirements.txt"
    )
    sys.exit(1)


class AIServiceStarter:
    """Handles startup logic for the AI service"""

    def __init__(self):
        # Load environment variables from .env file
        self._load_environment()
        self.logger = self._setup_logging()
        self.server = None

    def _load_environment(self) -> None:
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded environment from {env_file}")
        else:
            print(
                f"⚠️  No .env file found at {env_file}, using system environment variables"
            )

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                (
                    logging.FileHandler("logs/startup.log", mode="a")
                    if Path("logs").exists()
                    else logging.NullHandler()
                ),
            ],
        )
        return logging.getLogger(__name__)

    def validate_environment(self) -> bool:
        """Validate required environment variables and configuration"""
        self.logger.info("🔍 Validating environment...")

        # Check for API keys
        required_vars = ["GEMINI_API_KEY", "CLAUDE_API_KEY"]

        # Optional but recommended vars
        optional_vars = [
            "HOST",
            "PORT",
            "ENVIRONMENT",
            "LOG_LEVEL",
            "MAX_TOKENS",
            "TEMPERATURE",
        ]

        missing_required = []
        missing_optional = []

        for var_name in required_vars:
            if not os.getenv(var_name):
                missing_required.append(var_name)

        for var_name in optional_vars:
            if not os.getenv(var_name):
                missing_optional.append(var_name)

        if missing_required:
            self.logger.warning(
                f"⚠️  Missing required environment variables: {', '.join(missing_required)}"
            )
            self.logger.warning(
                "The service will start but some features may not work properly"
            )

        if missing_optional:
            self.logger.info(f"ℹ️  Using defaults for: {', '.join(missing_optional)}")

        if not missing_required:
            self.logger.info("✅ Environment validation passed")
            return True

        return False

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        self.logger.info("🔍 Checking dependencies...")

        # Package name mapping: display_name -> actual_import_name
        required_packages = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "google-generativeai": "google.generativeai",
            "anthropic": "anthropic",
            "chromadb": "chromadb",
            "pandas": "pandas",
            "numpy": "numpy",
            "python-dotenv": "dotenv",
        }

        missing_packages = []
        for display_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                self.logger.debug(f"✓ {display_name}")
            except ImportError:
                missing_packages.append(display_name)
                self.logger.debug(f"✗ {display_name}")

        if missing_packages:
            self.logger.error(
                f"❌ Missing required packages: {', '.join(missing_packages)}"
            )
            self.logger.error(
                "Please install them with: pip install -r requirements.txt"
            )
            return False

        self.logger.info("✅ All dependencies are available")
        return True

    def setup_directories(self) -> None:
        """Create required directories if they don't exist"""
        self.logger.info("📁 Setting up directories...")

        directories = ["logs", "chroma_db"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.debug(f"Created/verified directory: {directory}")

        self.logger.info("✅ Directories setup complete")

    async def health_check(self) -> bool:
        """Perform basic health checks"""
        self.logger.info("🏥 Performing health checks...")

        try:
            # Import app components after environment is validated
            from app.config.settings import settings
            from app.core.llm_client import llm_client

            # Test settings access
            _ = settings.HOST
            _ = settings.PORT

            self.logger.info("✅ Health checks passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Health check failed: {str(e)}")
            return False

    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers"""

        def signal_handler(signum, frame):
            self.logger.info(
                f"📡 Received signal {signum}, initiating graceful shutdown..."
            )
            if self.server:
                self.server.should_exit = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start_server(self, host: str, port: int, dev_mode: bool = False) -> None:
        """Start the FastAPI server"""
        self.logger.info("🚀 Starting AI Service...")

        # Import app after environment validation
        try:
            from app.main import app
        except ImportError as e:
            self.logger.error(f"❌ Failed to import FastAPI app: {e}")
            raise

        # Configure uvicorn
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level,
            reload=dev_mode,
            reload_dirs=["app"] if dev_mode else None,
            access_log=True,
            use_colors=True,
        )

        self.server = uvicorn.Server(config)

        try:
            self.logger.info(f"🌐 Server starting on http://{host}:{port}")
            if dev_mode:
                self.logger.info("🔄 Development mode: Auto-reload enabled")

            await self.server.serve()

        except Exception as e:
            self.logger.error(f"❌ Server startup failed: {str(e)}")
            raise

    async def run(self, host: str, port: int, dev_mode: bool = False) -> None:
        """Main startup sequence"""
        try:
            # Setup signal handlers for graceful shutdown
            self.setup_signal_handlers()

            # Create required directories
            self.setup_directories()

            # Validate environment
            env_valid = self.validate_environment()
            if not env_valid:
                self.logger.warning(
                    "⚠️  Starting with incomplete environment configuration"
                )

            # Check dependencies
            if not self.check_dependencies():
                self.logger.error("❌ Dependency check failed")
                sys.exit(1)

            # Perform health checks
            if not await self.health_check():
                self.logger.error("❌ Health check failed")
                sys.exit(1)

            # Start the server
            await self.start_server(host, port, dev_mode)

        except KeyboardInterrupt:
            self.logger.info("👋 Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"❌ Startup failed: {str(e)}")
            sys.exit(1)
        finally:
            self.logger.info("👋 AI Service stopped")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the AI Service")
    parser.add_argument(
        "--dev", action="store_true", help="Start in development mode with auto-reload"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "localhost"),
        help="Host to bind to (default: localhost or HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000 or PORT env var)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Set the logging level",
    )
    return parser.parse_args()


async def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()

    # Override LOG_LEVEL if provided via command line
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    # Print startup banner
    print("=" * 60)
    print("🤖 DocAnalyzer AI Service")
    print("=" * 60)
    print(f"🏠 Host: {args.host}")
    print(f"🚪 Port: {args.port}")
    print(f"🔧 Mode: {'Development' if args.dev else 'Production'}")
    print(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print("=" * 60)

    # Create and run the service
    starter = AIServiceStarter()
    await starter.run(args.host, args.port, args.dev)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
