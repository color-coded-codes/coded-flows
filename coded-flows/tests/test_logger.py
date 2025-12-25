import logging
import pytest
import tempfile
import os
from io import StringIO
from unittest.mock import patch
from coded_flows.utils import CodedFlowsLogger


class TestCodedFlowsLogger:
    """Test suite for CodedFlowsLogger class."""

    def test_logger_init_default(self):
        """Test logger initialization with default parameters."""
        logger = CodedFlowsLogger(name="TestApp")
        assert logger.logger.name == "TestApp"
        assert len(logger.logger.handlers) > 0

    def test_logger_init_custom_level(self):
        """Test logger initialization with custom level."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.DEBUG)
        console_handler = logger.logger.handlers[0]
        assert console_handler.level == logging.DEBUG

    def test_logger_init_with_file(self):
        """Test logger initialization with file logging."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            tmp_path = tmp.name

        try:
            logger = CodedFlowsLogger(name="TestApp", log_file=tmp_path)
            assert len(logger.logger.handlers) == 2  # Console + File

            # Write a log message
            logger.info("Test message")

            # Check file was created and contains the message
            with open(tmp_path, "r") as f:
                content = f.read()
                assert "Test message" in content
                assert "INFO" in content
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_logger_file_level(self):
        """Test that file handler uses specified level."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            tmp_path = tmp.name

        try:
            logger = CodedFlowsLogger(
                name="TestApp",
                level=logging.WARNING,  # Console: WARNING
                log_file=tmp_path,
                file_level=logging.DEBUG,  # File: DEBUG
            )

            # File handler should be the second handler
            file_handler = logger.logger.handlers[1]
            assert file_handler.level == logging.DEBUG
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_get_logger(self):
        """Test get_logger method returns correct logger instance."""
        logger = CodedFlowsLogger(name="TestApp")
        log_instance = logger.get_logger()
        assert isinstance(log_instance, logging.Logger)
        assert log_instance.name == "TestApp"

    def test_convenience_methods(self):
        """Test convenience logging methods."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.DEBUG)

        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_no_duplicate_handlers(self):
        """Test that creating multiple loggers doesn't create duplicate handlers."""
        logger1 = CodedFlowsLogger(name="TestApp")
        initial_handler_count = len(logger1.logger.handlers)

        # Creating another logger with same name should clear old handlers
        logger2 = CodedFlowsLogger(name="TestApp")
        assert len(logger2.logger.handlers) == initial_handler_count

    def test_custom_format(self):
        """Test logger with custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        logger = CodedFlowsLogger(name="TestApp", log_format=custom_format)

        # Check that formatter uses custom format
        handler = logger.logger.handlers[0]
        assert handler.formatter is not None

    def test_custom_date_format(self):
        """Test logger with custom date format."""
        custom_date_format = "%Y/%m/%d"
        logger = CodedFlowsLogger(name="TestApp", date_format=custom_date_format)

        handler = logger.logger.handlers[0]
        assert handler.formatter.datefmt == custom_date_format

    @patch("sys.stdout", new_callable=StringIO)
    def test_console_output(self, mock_stdout):
        """Test that messages are written to console."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.INFO)
        logger.info("Test console message")

        output = mock_stdout.getvalue()
        assert "Test console message" in output
        assert "INFO" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_console_output_has_colors(self, mock_stdout):
        """Test that console output contains ANSI color codes."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.INFO)
        logger.info("Colored message")

        output = mock_stdout.getvalue()
        # Should contain ANSI escape codes for colors
        assert "\033[" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_level_filtering(self, mock_stdout):
        """Test that log level filtering works."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.WARNING)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        output = mock_stdout.getvalue()

        # Only WARNING and above should appear
        assert "Debug message" not in output
        assert "Info message" not in output
        assert "Warning message" in output

    def test_file_without_colors(self):
        """Test that file output doesn't contain color codes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as tmp:
            tmp_path = tmp.name

        try:
            logger = CodedFlowsLogger(name="TestApp", log_file=tmp_path)
            logger.info("Test message")

            with open(tmp_path, "r") as f:
                content = f.read()
                # Should NOT contain ANSI escape codes
                assert "\033[" not in content
                assert "Test message" in content
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration:
    """Integration tests for the logging system."""

    def test_multiple_loggers_different_names(self):
        """Test creating multiple loggers with different names."""
        logger1 = CodedFlowsLogger(name="App1")
        logger2 = CodedFlowsLogger(name="App2")

        assert logger1.logger.name == "App1"
        assert logger2.logger.name == "App2"
        assert logger1.logger != logger2.logger

    @patch("sys.stdout", new_callable=StringIO)
    def test_formatted_output_structure(self, mock_stdout):
        """Test that output follows expected structure."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.INFO)
        logger.info("Test message")

        output = mock_stdout.getvalue()

        # Should contain: timestamp | level | name | message
        assert "|" in output
        parts = output.split("|")
        assert len(parts) >= 4  # At least 4 parts

    def test_logger_with_arguments(self):
        """Test logging with format arguments."""
        logger = CodedFlowsLogger(name="TestApp", level=logging.DEBUG)

        # Should not raise exceptions
        logger.info("User %s logged in", "john")
        logger.debug("Processing %d items", 42)
