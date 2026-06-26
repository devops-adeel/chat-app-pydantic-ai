import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// ExperimentalEmptyAdapter is required — CopilotKit Runtime won't initialise
// without a service adapter. The empty one is correct here because pydantic-ai
// handles all LLM calls; the runtime is only a protocol bridge.
const serviceAdapter = new ExperimentalEmptyAdapter();

const pydanticAgent = new HttpAgent({
  url: process.env.AGENT_URL ?? "http://localhost:8000/pydantic-agent",
});

const runtime = new CopilotRuntime({
  agents: { pydanticAgent },
});

const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  serviceAdapter,
  endpoint: "/api/copilotkit",
});

export const POST = (req: NextRequest) => handleRequest(req);
