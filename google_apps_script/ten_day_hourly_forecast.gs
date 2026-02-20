const GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search";
const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const HUGGINGFACE_OPENAI_COMPAT_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions";

const CONFIG = {
  zipCode: "39465",
  recipientEmail: "matthew.hudson@maatenergy.com",
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
  triggerHourLocal: 6,
  enableLlmSummary: true,
  hfModel: "meta-llama/Llama-3.2-3B-Instruct",
  hfTokenPropertyName: "HF_TOKEN",
  llmMaxRows: 240
};

function sendTenDayHourlyForecastEmail() {
  assertConfig();
  const geocoding = geocodeZipCode_(CONFIG.zipCode);
  const dateRange = buildDateRange_();
  const forecast = fetchForecast_(
    geocoding.latitude,
    geocoding.longitude,
    dateRange.startDate,
    dateRange.endDate
  );
  const csv = buildHourlyCsv_(forecast, CONFIG.hourlyMetrics);
  const fileName = "hourly_forecast_" + dateRange.startDate + "_to_" + dateRange.endDate + ".csv";
  const attachments = [Utilities.newBlob(csv, "text/csv", fileName)];
  let llmSummaryText = "";
  let llmSummaryForEmail = "";
  if (CONFIG.enableLlmSummary) {
    llmSummaryText = generateExtremeWeatherSummary_(forecast, geocoding, dateRange);
    llmSummaryForEmail = sanitizeLlmSummaryForEmail_(llmSummaryText);
    const summaryFileName = "llm_extreme_weather_summary_" + dateRange.startDate + "_to_" + dateRange.endDate + ".txt";
    attachments.push(Utilities.newBlob(llmSummaryForEmail + "\n", "text/plain", summaryFileName));
  }

  const subject = "10-day hourly forecast: " + geocoding.label + " (" + dateRange.startDate + ")";
  let body =
    "Location: " + geocoding.label + "\n" +
    "Coordinates: " + geocoding.latitude + ", " + geocoding.longitude + "\n" +
    "Range: " + dateRange.startDate + " to " + dateRange.endDate + "\n" +
    "Metrics: " + CONFIG.hourlyMetrics.join(", ") + "\n\n" +
    "Attached: CSV with hourly forecast data from Open-Meteo.";
  if (llmSummaryForEmail) {
    body += "\n\nWeather Newsletter\n" + llmSummaryForEmail;
  }

  GmailApp.sendEmail(CONFIG.recipientEmail, subject, body, {
    attachments: attachments,
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
  if (!CONFIG.zipCode || !CONFIG.recipientEmail || !CONFIG.timezone) {
    throw new Error("Set CONFIG.zipCode, CONFIG.recipientEmail, and CONFIG.timezone.");
  }
  if (!CONFIG.hourlyMetrics || CONFIG.hourlyMetrics.length === 0) {
    throw new Error("Set at least one hourly metric in CONFIG.hourlyMetrics.");
  }
  if (CONFIG.enableLlmSummary && (!CONFIG.hfModel || !CONFIG.hfTokenPropertyName)) {
    throw new Error("Set CONFIG.hfModel and CONFIG.hfTokenPropertyName when enableLlmSummary is true.");
  }
  if (CONFIG.enableLlmSummary && (!CONFIG.llmMaxRows || CONFIG.llmMaxRows <= 0)) {
    throw new Error("Set CONFIG.llmMaxRows to a positive integer.");
  }
}

function geocodeZipCode_(zipCode) {
  const params = {
    name: zipCode,
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
    throw new Error("No geocoding results for ZIP code: " + zipCode);
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

function generateExtremeWeatherSummary_(forecastPayload, geocoding, dateRange) {
  const prompt = buildLlmPromptFromForecast_(forecastPayload, geocoding.label, dateRange, CONFIG.llmMaxRows);
  const hfToken = getRequiredScriptProperty_(CONFIG.hfTokenPropertyName);
  const response = UrlFetchApp.fetch(HUGGINGFACE_OPENAI_COMPAT_CHAT_COMPLETIONS_URL, {
    method: "post",
    contentType: "application/json",
    headers: {
      Authorization: "Bearer " + hfToken
    },
    payload: JSON.stringify({
      model: CONFIG.hfModel,
      messages: [
        {
          role: "system",
          content:
            "You are a meteorological analyst writing a local weather newsletter. " +
            "Use plain text only and never use markdown characters for formatting. " +
            "Focus on notable weather patterns and anomalies. Do not invent values. " +
            "If confidence is limited, state uncertainty explicitly."
        },
        { role: "user", content: prompt }
      ],
      max_tokens: 700,
      temperature: 0.1
    }),
    muteHttpExceptions: true
  });

  if (response.getResponseCode() < 200 || response.getResponseCode() >= 300) {
    let detail = "";
    try {
      detail = extractHfErrorMessage_(JSON.parse(response.getContentText()));
    } catch (e) {
      detail = "";
    }
    throw new Error(
      "Hugging Face request failed. HTTP " +
        response.getResponseCode() +
        (detail ? ": " + detail : "")
    );
  }

  const payload = JSON.parse(response.getContentText());
  const text = extractHfGeneratedText_(payload);
  if (!text) {
    throw new Error("Hugging Face response did not include generated summary text.");
  }
  return text;
}

function buildLlmPromptFromForecast_(forecastPayload, locationLabel, dateRange, maxRows) {
  const hourly = forecastPayload.hourly || {};
  const times = hourly.time || [];
  const maxRowCount = Math.min(maxRows, times.length);
  const lines = [
    "Location: " + locationLabel,
    "Date range: " + dateRange.startDate + " to " + dateRange.endDate,
    "Task: Identify notable and potentially extreme weather events from the CSV data.",
    "Temperature data in this CSV is in degrees Fahrenheit (F). Do not convert temperature values to Celsius.",
    "Units: temperature_2m=F, apparent_temperature=F, relative_humidity_2m=%, precipitation=inch, wind_speed_10m=mph, cloud_cover=%.",
    "Write an engaging plain-text newsletter for local readers.",
    "Return sections titled exactly: Summary:, Notable Weather Moments:, Impacts and Risks:, and What to Watch Next:.",
    "Do not use markdown. Do not use #, *, **, or bullet syntax.",
    "Use concrete timestamps in AM/PM format and values whenever possible.",
    "",
    summarizeForecastExtremes_(hourly, CONFIG.hourlyMetrics),
    "",
    "hourly CSV sample (first " + maxRowCount + " rows):",
    buildHourlyCsvFromSeries_(hourly, CONFIG.hourlyMetrics, maxRowCount)
  ];
  return lines.join("\n");
}

function summarizeForecastExtremes_(hourly, metrics) {
  const parts = ["hourly extreme snapshots:"];
  for (let i = 0; i < metrics.length; i += 1) {
    const metric = metrics[i];
    const series = hourly[metric] || [];
    const times = hourly.time || [];
    let minVal = null;
    let maxVal = null;
    let minTime = "n/a";
    let maxTime = "n/a";
    for (let j = 0; j < series.length; j += 1) {
      const raw = series[j];
      if (raw === null || raw === undefined || raw === "") {
        continue;
      }
      const value = Number(raw);
      if (!isFinite(value)) {
        continue;
      }
      if (minVal === null || value < minVal) {
        minVal = value;
        minTime = times[j] || "n/a";
      }
      if (maxVal === null || value > maxVal) {
        maxVal = value;
        maxTime = times[j] || "n/a";
      }
    }
    if (minVal === null || maxVal === null) {
      continue;
    }
    parts.push(
      "- " +
        metric +
        ": min=" +
        formatValueWithUnit_(metric, minVal) +
        " at " +
        minTime +
        ", max=" +
        formatValueWithUnit_(metric, maxVal) +
        " at " +
        maxTime
    );
  }
  return parts.join("\n");
}

function formatValueWithUnit_(metric, value) {
  const unit = getMetricUnit_(metric);
  if (!isFinite(value)) {
    return String(value);
  }
  return value.toFixed(2) + (unit ? " " + unit : "");
}

function getMetricUnit_(metric) {
  if (metric === "temperature_2m" || metric === "apparent_temperature") {
    return "F";
  }
  if (metric === "relative_humidity_2m" || metric === "cloud_cover") {
    return "%";
  }
  if (metric === "precipitation") {
    return "inch";
  }
  if (metric === "wind_speed_10m") {
    return "mph";
  }
  return "";
}

function sanitizeLlmSummaryForEmail_(text) {
  if (!text) {
    return "";
  }
  let normalized = String(text).replace(/\r/g, "");
  normalized = normalized.replace(/```/g, "");
  normalized = normalized.replace(/^#{1,6}\s+/gm, "");
  normalized = normalized.replace(/\*\*(.*?)\*\*/g, "$1");
  normalized = normalized.replace(/\*(.*?)\*/g, "$1");
  normalized = normalized.replace(/^\s*[-*+]\s+/gm, "- ");
  normalized = normalized.replace(/^\s*\d+\.\s+/gm, "- ");
  normalized = normalized.replace(/\n{3,}/g, "\n\n");
  return normalized.trim();
}

function buildHourlyCsvFromSeries_(hourly, metrics, maxRows) {
  const header = ["time"].concat(metrics);
  const rows = [header];
  const rowCount = Math.min(maxRows, (hourly.time || []).length);
  for (let i = 0; i < rowCount; i += 1) {
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

function getRequiredScriptProperty_(name) {
  const value = PropertiesService.getScriptProperties().getProperty(name);
  if (!value) {
    throw new Error("Missing Script Property '" + name + "'. Set it before running LLM summary.");
  }
  return value;
}

function extractHfGeneratedText_(payload) {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  if (payload.choices && payload.choices.length > 0) {
    const first = payload.choices[0];
    if (first.message && typeof first.message.content === "string") {
      return first.message.content.trim();
    }
    if (typeof first.text === "string") {
      return first.text.trim();
    }
  }
  if (typeof payload.generated_text === "string") {
    return payload.generated_text.trim();
  }
  return "";
}

function extractHfErrorMessage_(payload) {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  if (typeof payload.error === "string") {
    return payload.error.trim();
  }
  if (typeof payload.message === "string") {
    return payload.message.trim();
  }
  return "";
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
