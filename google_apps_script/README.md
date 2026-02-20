# Google Apps Script: Daily 10-Day Hourly Forecast Email

## What this does

The script in `google_apps_script/ten_day_hourly_forecast.gs`:

- Geocodes your location with Open-Meteo
- Pulls hourly forecast data for today through today + 9 days
- Builds a CSV attachment
- Emails it to your Google Workspace address
- Runs daily using a time-based trigger

## Setup

1. Open [script.google.com](https://script.google.com) and create a new project.
2. Replace the default script with the contents of `google_apps_script/ten_day_hourly_forecast.gs`.
3. Edit the `CONFIG` block:
   - `locationQuery` (example: `Petal, United States`)
   - `recipientEmail` (your Google Workspace mailbox)
   - `timezone` (example: `America/Chicago`)
   - `hourlyMetrics` (the columns you want)
   - `triggerHourLocal` (0-23 local hour)
4. Save the project.
5. Run `sendTenDayHourlyForecastEmail` once from the editor to authorize permissions.
6. Run `createDailyForecastTrigger` once to install the daily schedule.

## Operational notes

- Re-run `createDailyForecastTrigger` after changing `triggerHourLocal`.
- Run `deleteDailyForecastTriggers` to remove the automation.
- Open-Meteo forecast is free and does not require an API key.
