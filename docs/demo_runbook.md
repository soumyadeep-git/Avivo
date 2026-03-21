# Demo Runbook

## Goal

Show the bot behaving like a production-ready Telegram assistant rather than a toy prototype.

## Demo checklist

1. Start the bot
2. Show `/start`
3. Show `/help`
4. Ask a grounded documentation question
5. Ask a follow-up question
6. Trigger `/summarize`
7. Upload an image
8. Show health endpoint output
9. Run the benchmark script

## Suggested prompt sequence

1. `/start`
2. `/ask What are path parameters in FastAPI?`
3. `/ask How are query parameters different?`
4. `/summarize`
5. Upload a screenshot or photo

## What to point out

- The answer includes evidence status
- Source citations are shown explicitly
- Retrieved snippet previews are visible
- The app supports both text and image flows
- The project has a webhook deployment path, health endpoints, tests, and a benchmark script
