# TripMate AI — Multi-Agent Travel Planning Assistant

This sample shows how to build a **multi-agent travel planning assistant** using **Azure AI Foundry** and **Azure Developer CLI (`azd`)**, showcasing three different agent types:

| Agent | Type | Container? | Framework |
|-------|------|-----------|-----------|
| `travel-concierge` | **Declarative Prompt Agent** | No | None (SDK-defined via `PromptAgentDefinition`) |
| `trip-scout` | **Hosted Agent** | Yes | agent-framework |
| `booking-manager` | **Hosted Agent** | Yes | LangGraph |
| `tripmate` | **Workflow** | No | Foundry Workflow YAML |

A single `azd up`:

- Creates the Azure AI Foundry project, Bing Custom Search, and supporting resources using Bicep
- Builds and pushes agent container images to Azure Container Registry (ACR)
- Creates a declarative prompt agent (travel concierge) — no container needed
- Creates hosted agents for trip search and booking management
- Deploys a workflow that orchestrates the full conversation

## Scenario

```
User: "I want to plan a weekend trip to Barcelona in June"
  → Travel Concierge (prompt agent) classifies as trip-scout
  → Trip Scout searches flights, hotels, activities → returns options

User: "Book the Hilton and the morning Lufthansa flight"
  → Travel Concierge classifies as booking-manager
  → Booking Manager checks availability → confirms booking → returns itinerary

User: "Can I change my hotel to the W?"
  → Travel Concierge classifies as booking-manager
  → Booking Manager looks up booking → modifies → returns updated itinerary

User: "Hello!"
  → Travel Concierge classifies as none → responds directly with a greeting
```

## Prerequisites

- Azure subscription with permissions to create resources
- [Azure Developer CLI (`azd`)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
- [Azure CLI (`az`)](https://learn.microsoft.com/cli/azure/install-azure-cli) (used by the postdeploy script)
- Docker (for local image builds in ACR)
- Python 3.10+ (for the deployment script and agents)

## Quick Start: Provision + Deploy with `azd up`

1. **Login to Azure:**

   ```bash
   azd auth login
   ```

2. **Initialize the environment (optional but recommended):**

   ```bash
   azd init
   # When prompted, choose a new or existing environment name
   ```

3. **Provision infrastructure and deploy agents:**

   From the repo root:

   ```bash
   azd up
   ```

   During `azd up` the following happens:

   - Bicep templates in `infra/` are deployed (AI project, ACR, Bing Custom Search, storage, monitoring, etc.).
   - `azd` writes an environment-specific `.env` file into `.azure/<env-name>/.env` with outputs like `AZURE_AI_PROJECT_ENDPOINT`, `AZURE_CONTAINER_REGISTRY_ENDPOINT`, `BING_CUSTOM_GROUNDING_CONNECTION_NAME`, etc.
   - The **postdeploy hooks** defined in `azure.yaml` run automatically (see below).

4. **Verify deployment:**

   After `azd up` completes successfully, you should have:

   - A `.env` file at the repo root populated with connection info and image URLs
   - Four agents created in your Azure AI project:
     - `travel-concierge` – Declarative prompt agent that classifies intent and routes
     - `trip-scout` – Searches flights, hotels, and activities via Bing
     - `booking-manager` – Handles booking, modification, and cancellation
     - `tripmate` – Workflow that orchestrates the conversation

## Included Agents

### Travel Concierge (Declarative Prompt Agent)

Defined entirely via `PromptAgentDefinition` in `deploy_agents.py` — no container, no Dockerfile. Uses a system prompt with `TextResponseFormatJsonSchema` to classify user intent into structured JSON and route to the appropriate specialist agent. Can also respond directly to greetings and general travel questions.

### Trip Scout (agent-framework Hosted Agent)

An agent-framework-based container agent that searches for flights, hotels, and activities based on user travel queries. Uses Bing Custom Search for grounding. Returns structured results with prices, ratings, and an estimated trip budget.

### Booking Manager (LangGraph Hosted Agent)

A LangGraph-based container agent with a stateful tool-calling loop. Provides five booking tools: `check_availability`, `create_booking`, `modify_booking`, `get_booking`, and `cancel_booking`. The LLM decides which tools to call, executes them, and loops until the booking flow is complete.

### TripMate Workflow

A Foundry Workflow YAML (`tripmate.yaml`) that orchestrates the conversation:
1. Invokes the travel concierge to classify user intent
2. Conditionally routes to trip-scout, booking-manager, or responds directly

## How the `postdeploy` Hook Works

The `postdeploy` section in `azure.yaml` looks like this:

```yaml
hooks:
  postdeploy:
    - name: copy-env-file
      description: Copy environment file from azd environment folder
      shell: sh
      run: cp .azure/$AZURE_ENV_NAME/.env ./
    - name: build-and-push-container-images
      description: Build and push container images to ACR
      shell: sh
      run: ./scripts/postdeploy.sh trip-scout:./src/agents/trip-scout booking-manager:./src/agents/booking-manager
```

In order, it does:

1. **`copy-env-file`** – Copies the `.env` generated by `azd` for the current environment to the repo root. This makes variables like `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_CONTAINER_REGISTRY_ENDPOINT` available to local tools and scripts.
2. **`build-and-push-container-images`** – Calls `scripts/postdeploy.sh` with one or more `<image-name>:<path>` arguments.

   For each argument, `scripts/postdeploy.sh`:
   - Uses `AZURE_CONTAINER_REGISTRY_ENDPOINT` from the `.env` file to determine which ACR to use
   - Builds and pushes an image to ACR via `az acr build`
   - Writes an environment variable of the form `<IMAGE_NAME>_IMAGE=<full-image-tag>` into the root `.env` file
   - After all images are built, runs `python deploy_agents.py` in `src/`

3. **`src/deploy_agents.py`** – Reads the `.env` file (via `python-dotenv`) and:
   - Creates the **travel concierge** as a `PromptAgentDefinition` (declarative prompt agent — no container)
   - Discovers all `*_IMAGE` variables (e.g., `TRIP_SCOUT_IMAGE`, `BOOKING_MANAGER_IMAGE`)
   - Creates or updates a **hosted agent** for each container image
   - Configures Bing Custom Search tool integration when `BING_CUSTOM_GROUNDING_CONNECTION_NAME` is set
   - Deploys all workflow YAML files from `src/workflows/` as **workflow agents**

Net result: every time you run `azd up` (or `azd deploy` that triggers the postdeploy hooks), your hosted agents are rebuilt and redeployed from source.

## Project Structure (Relevant Parts)

```text
azure.yaml                 # azd configuration + postdeploy hooks
infra/                     # Bicep templates for infra + AI project
scripts/
  postdeploy.sh            # Builds images, pushes to ACR, runs deploy_agents.py
src/
  deploy_agents.py         # Creates all agents: prompt, hosted, and workflow
  agents/
    trip-scout/
      agent.py             # Travel search agent (agent-framework)
      Dockerfile           # Container definition
    booking-manager/
      agent.py             # Booking management agent (LangGraph)
      Dockerfile           # Container definition
  config/
    settings.py            # Helper for reading config from env
  workflows/
    tripmate.yaml          # TripMate workflow orchestration
```

## Customizing Agents and Images

- **Add another hosted agent**
  - Create a new folder under `src/agents/<your-agent-name>/` with an `agent.py` and `Dockerfile`.
  - Update the `build-and-push-container-images` hook in `azure.yaml`, passing an additional argument, e.g.:

    ```yaml
    run: ./scripts/postdeploy.sh \
      trip-scout:./src/agents/trip-scout \
      booking-manager:./src/agents/booking-manager \
      my-new-agent:./src/agents/my-new-agent
    ```

  - Re-run:

    ```bash
    azd up
    # or, after infra exists
    azd deploy
    ```

- **Change model or project endpoint**
  - The script `src/deploy_agents.py` reads `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME` from the environment (`.env` file).
  - `AZURE_AI_PROJECT_ENDPOINT` is provided by the Bicep deployment and `azd`.
  - You can override `AZURE_AI_MODEL_DEPLOYMENT_NAME` in the root `.env` if needed (defaults to `o4-mini`).

## Running the Deployment Script Manually (Optional)

If you change only the container images and want to re-register agents without re-running full `azd up`:

1. Ensure your root `.env` is up to date and contains `*_IMAGE` variables.
2. From the repo root, run:

   ```bash
   cd src
   python deploy_agents.py
   ```

This will re-create or update the hosted agents using the already-pushed images.

## SDK Compatibility Notes

This sample targets **`azure-ai-projects` 2.0.1** (stable). It uses:

- `PromptAgentDefinition` — declarative prompt agents (GA, no opt-in needed)
- `HostedAgentDefinition` — containerized hosted agents (requires `FoundryFeaturesOptInKeys.HOSTED_AGENTS_V1_PREVIEW`)
- `WorkflowAgentDefinition` — workflow agents (requires `FoundryFeaturesOptInKeys.WORKFLOW_AGENTS_V1_PREVIEW`)

## Hosted Agents: Networking Limitations (Preview)

Hosted Agents (containerized custom code) have the following networking restrictions during preview:

- **Hosted agents do NOT support private networking** with Standard Setup.
- **They cannot be deployed into network-isolated Foundry environments.**

This is explicitly documented as a preview limitation — see the [Azure AI Foundry networking docs](https://learn.microsoft.com/azure/ai-foundry/concepts/network-security-overview).

### Networking Summary

| Area | Status |
|---|---|
| BYO VNet with Private Endpoints | ✅ GA |
| No public egress | ✅ GA |
| Access to private Azure PaaS | ✅ GA |
| On-prem via VPN / ExpressRoute | ✅ GA |
| Cross-region VNet | ❌ Not supported |
| Inbound private IP to agent runtime | ❌ Not supported |
| Managed VNet | ⚠️ Preview |
| **Hosted agents + private networking** | **❌ Not supported** |

## Cleanup

To remove all provisioned resources:

```bash
azd down
```

This deletes the resource group created by `azd` (including the AI project, ACR, and other resources).
