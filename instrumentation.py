from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic_ai import Agent


def configure_telemetry() -> None:
    """
    Wire up OTEL for pydantic-ai traces.

    PoC mode: ConsoleSpanExporter — spans appear in stdout with no external service.

    Langfuse production mode: replace the two lines marked SWAP with:

        import base64, os
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        pk = os.environ["LANGFUSE_PUBLIC_KEY"]
        sk = os.environ["LANGFUSE_SECRET_KEY"]
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {base64.b64encode(f'{pk}:{sk}'.encode()).decode()}"},
        )))
    """
    provider = TracerProvider()
    # SWAP: replace these two lines for Langfuse production
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    # END SWAP
    trace.set_tracer_provider(provider)

    # Instrument all Agent instances globally — must call after set_tracer_provider
    Agent.instrument_all()
