const GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search";
const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";

const CONFIG = {
  locationQuery: "Petal, United States",
  recipientEmail: "you@yourdomain.com",
  timezone: "America/Chicago",
  hourlyMetrics: [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover"
  ],
  temperatureUnit: "fahrenheit",
  windSpeedUnit: "mph",
  precipitationUnit: "inch",
  triggerHourLocal: 6
};

function sendTenDayHourlyForecastEmail() {
  assertConfig();
  const geocoding = geocodeLocation_(CONFIG.locationQuery);
  const dateRange = buildDateRange_();
  const forecast = fetchForecast_(
    geocoding.latitude,
    geocoding.longitude,
    dateRange.startDate,
    dateRange.endDate
  );
  const csv = buildHourlyCsv_(forecast, CONFIG.hourlyMetrics);
  const fileName = "hourly_forecast_" + dateRange.startDate + "_to_" + dateRange.endDate + ".csv";
  const attachment = Utilities.newBlob(csv, "text/csv", fileName);

  const subject = "10-day hourly forecast: " + geocoding.label + " (" + dateRange.startDate + ")";
  const body =
    "Location: " + geocoding.label + "\n" +
    "Coordinates: " + geocoding.latitude + ", " + geocoding.longitude + "\n" +
    "Range: " + dateRange.startDate + " to " + dateRange.endDate + "\n" +
    "Metrics: " + CONFIG.hourlyMetrics.join(", ") + "\n\n" +
    "Attached: CSV with hourly forecast data from Open-Meteo.";

  GmailApp.sendEmail(CONFIG.recipientEmail, subject, body, {
    attachments: [attachment],
    name: "Weather Forecast Bot"
  });
}

function createDailyForecastTrigger() {
  deleteDailyForecastTriggers();
  ScriptApp.newTrigger("sendTenDayHourlyForecastEmail")
    .timeBased()
    .everyDays(1)
    .atHour(CONFIG.triggerHourLocal)
    .create();
}

function deleteDailyForecastTriggers() {
  const triggers = ScriptApp.getProjectTriggers();
  for (let i = 0; i < triggers.length; i += 1) {
    if (triggers[i].getHandlerFunction() === "sendTenDayHourlyForecastEmail") {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
}

function assertConfig() {
  if (!CONFIG.locationQuery || !CONFIG.recipientEmail || !CONFIG.timezone) {
    throw new Error("Set CONFIG.locationQuery, CONFIG.recipientEmail, and CONFIG.timezone.");
  }
  if (!CONFIG.hourlyMetrics || CONFIG.hourlyMetrics.length === 0) {
    throw new Error("Set at least one hourly metric in CONFIG.hourlyMetrics.");
  }
}

function geocodeLocation_(locationQuery) {
  const params = {
    name: locationQuery,
    count: 1,
    language: "en",
    format: "json"
  };
  const response = UrlFetchApp.fetch(GEOCODING_URL + "?" + toQueryString_(params), {
    muteHttpExceptions: true
  });
  if (response.getResponseCode() !== 200) {
    throw new Error("Geocoding failed. HTTP " + response.getResponseCode() + ": " + response.getContentText());
  }

  const payload = JSON.parse(response.getContentText());
  if (!payload.results || payload.results.length === 0) {
    throw new Error("No geocoding results for location query: " + locationQuery);
  }

  const hit = payload.results[0];
  return {
    latitude: hit.latitude,
    longitude: hit.longitude,
    label: [hit.name, hit.country].filter(Boolean).join(", ")
  };
}

function buildDateRange_() {
  const now = new Date();
  const startDate = Utilities.formatDate(now, CONFIG.timezone, "yyyy-MM-dd");
  const end = new Date(now.getTime());
  end.setDate(end.getDate() + 9);
  const endDate = Utilities.formatDate(end, CONFIG.timezone, "yyyy-MM-dd");
  return { startDate: startDate, endDate: endDate };
}

function fetchForecast_(latitude, longitude, startDate, endDate) {
  const params = {
    latitude: latitude,
    longitude: longitude,
    timezone: CONFIG.timezone,
    start_date: startDate,
    end_date: endDate,
    hourly: CONFIG.hourlyMetrics.join(","),
    temperature_unit: CONFIG.temperatureUnit,
    wind_speed_unit: CONFIG.windSpeedUnit,
    precipitation_unit: CONFIG.precipitationUnit
  };

  const response = UrlFetchApp.fetch(FORECAST_URL + "?" + toQueryString_(params), {
    muteHttpExceptions: true
  });
  if (response.getResponseCode() !== 200) {
    throw new Error("Forecast request failed. HTTP " + response.getResponseCode() + ": " + response.getContentText());
  }

  const payload = JSON.parse(response.getContentText());
  if (!payload.hourly || !payload.hourly.time) {
    throw new Error("Forecast payload missing hourly.time.");
  }
  return payload;
}

function buildHourlyCsv_(forecastPayload, metrics) {
  const hourly = forecastPayload.hourly;
  const header = ["time"].concat(metrics);
  const rows = [header];
  for (let i = 0; i < hourly.time.length; i += 1) {
    const row = [hourly.time[i]];
    for (let j = 0; j < metrics.length; j += 1) {
      const metric = metrics[j];
      const series = hourly[metric] || [];
      row.push(series[i]);
    }
    rows.push(row);
  }
  return rows.map(csvRow_).join("\n");
}

function csvRow_(cells) {
  return cells.map(csvCell_).join(",");
}

function csvCell_(value) {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value);
  if (text.indexOf(",") >= 0 || text.indexOf("\"") >= 0 || text.indexOf("\n") >= 0) {
    return "\"" + text.replace(/"/g, "\"\"") + "\"";
  }
  return text;
}

function toQueryString_(params) {
  const pairs = [];
  const keys = Object.keys(params);
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    pairs.push(encodeURIComponent(key) + "=" + encodeURIComponent(params[key]));
  }
  return pairs.join("&");
}
