# Google Apps Script: Daily 10-Day Hourly Forecast Email

## What this does

The script in `google_apps_script/ten_day_hourly_forecast.gs`:

- Geocodes your ZIP/postal code with Open-Meteo
- Pulls hourly forecast data for today through today + 9 days
- Builds a CSV attachment
- Optionally generates an LLM extreme-weather summary with Hugging Face
- Emails it to your Google Workspace address
- Runs daily using a time-based trigger

## Setup

1. Open [script.google.com](https://script.google.com) and create a new project.
2. Replace the default script with the contents of `google_apps_script/ten_day_hourly_forecast.gs`.
3. Edit the `CONFIG` block:
   - `zipCode` (example: `39465`)
   - `recipientEmail` (your Google Workspace mailbox)
   - `timezone` (example: `America/Chicago`)
   - `hourlyMetrics` (the columns you want)
   - `triggerHourLocal` (0-23 local hour)
   - Optional LLM settings:
     - `enableLlmSummary` (`true` to enable)
     - `hfModel` (example: `mistralai/Mistral-7B-Instruct-v0.3`)
     - `hfTokenPropertyName` (Script Property key, default `HF_TOKEN`)
     - `llmMaxRows` (max rows included in LLM prompt)
4. If `enableLlmSummary` is `true`, set your Hugging Face token in Script Properties:
   - Apps Script editor: Project Settings -> Script Properties -> Add property
   - Key: `HF_TOKEN` (or your `hfTokenPropertyName`)
   - Value: your Hugging Face token
5. Save the project.
6. Run `sendTenDayHourlyForecastEmail` once from the editor to authorize permissions.
7. Run `createDailyForecastTrigger` once to install the daily schedule.

## Operational notes

- Re-run `createDailyForecastTrigger` after changing `triggerHourLocal`.
- Run `deleteDailyForecastTriggers` to remove the automation.
- Open-Meteo forecast is free and does not require an API key.
