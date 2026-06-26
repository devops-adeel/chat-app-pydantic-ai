"use client";

import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="pydanticAgent">
      <main className="flex h-full flex-col">
        <CopilotChat
          labels={{
            title: "Personal Assistant",
            initial:
              "Hi! I can search the web to help you. What would you like to know?",
          }}
          className="flex-1"
        />
      </main>
    </CopilotKit>
  );
}
