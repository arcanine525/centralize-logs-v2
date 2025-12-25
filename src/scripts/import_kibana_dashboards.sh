#!/usr/bin/env bash
# ============================================================================
# Import Kibana Dashboards
# Creates Data Views and imports dashboards for log analysis and DDoS detection
# ============================================================================

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
ES_URL="${ES_URL:-http://localhost:9200}"

echo "========================================"
echo "Kibana Dashboard Import Script"
echo "========================================"
echo "Kibana URL: $KIBANA_URL"
echo "ES URL: $ES_URL"
echo ""

# Wait for Kibana to be ready
echo "Waiting for Kibana to be ready..."
until curl -sf "$KIBANA_URL/api/status" > /dev/null 2>&1; do
    echo "  Kibana not ready, waiting..."
    sleep 5
done
echo "Kibana is ready!"

# ============================================================================
# Create Data Views (Index Patterns)
# ============================================================================

echo ""
echo "Creating Data Views..."

# Web Logs Data View (use explicit id to match dashboard references)
echo "  Creating web-logs-* data view..."
curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" \
  -H "kbn-xsrf: true" \
  -H "Content-Type: application/json" \
  -d '{
    "data_view": {
      "id": "web-logs-*",
      "title": "web-logs-*",
      "name": "Web Logs",
      "timeFieldName": "@timestamp"
    }
  }' > /dev/null 2>&1 || echo "    (already exists or error)"

# DDoS Logs Data View (use explicit id to match dashboard references)
echo "  Creating ddos-logs data view..."
curl -sf -X POST "$KIBANA_URL/api/data_views/data_view" \
  -H "kbn-xsrf: true" \
  -H "Content-Type: application/json" \
  -d '{
    "data_view": {
      "id": "ddos-logs",
      "title": "ddos-logs",
      "name": "DDoS Detection Logs",
      "timeFieldName": "timestamp"
    }
  }' > /dev/null 2>&1 || echo "    (already exists or error)"

echo "Data Views created!"

# ============================================================================
# Import Dashboard
# ============================================================================

echo ""
echo "Importing dashboards..."

# Create dashboard export file
DASHBOARD_FILE="/tmp/kibana_dashboards.ndjson"

# Dashboard 1: Log Analysis Dashboard with 4 panels
# Panel 1: Request Timeline (Line Chart) - split by status_code, 1 minute interval
# Panel 2: Status Code Distribution (Pie Chart)
# Panel 3: Top 10 IPs (Data Table)
# Panel 4: Geographic Distribution (Bar Chart by Country)

cat > "$DASHBOARD_FILE" << 'DASHBOARD_EOF'
{"attributes":{"description":"Web traffic analysis with request timeline, status codes, top IPs and geographic distribution","kibanaSavedObjectMeta":{"searchSourceJSON":"{}"},"optionsJSON":"{\"useMargins\":true,\"syncColors\":true,\"syncCursor\":true,\"syncTooltips\":true,\"hidePanelTitles\":false}","panelsJSON":"[{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":0,\"y\":0,\"w\":48,\"h\":12,\"i\":\"panel-1\"},\"panelIndex\":\"panel-1\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Request Timeline\",\"description\":\"Requests over time split by status code\",\"visualizationType\":\"lnsXY\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"web-logs-*\"}],\"state\":{\"visualization\":{\"legend\":{\"isVisible\":true,\"position\":\"right\"},\"valueLabels\":\"hide\",\"fittingFunction\":\"None\",\"axisTitlesVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"tickLabelsVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"labelsOrientation\":{\"x\":0,\"yLeft\":0,\"yRight\":0},\"gridlinesVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"preferredSeriesType\":\"line\",\"layers\":[{\"layerId\":\"layer1\",\"accessors\":[\"col-count\"],\"position\":\"top\",\"seriesType\":\"line\",\"showGridlines\":false,\"layerType\":\"data\",\"xAccessor\":\"col-timestamp\",\"splitAccessor\":\"col-status\"}]},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col-timestamp\":{\"label\":\"@timestamp\",\"dataType\":\"date\",\"operationType\":\"date_histogram\",\"sourceField\":\"@timestamp\",\"isBucketed\":true,\"scale\":\"interval\",\"params\":{\"interval\":\"1m\",\"includeEmptyRows\":true,\"dropPartials\":false}},\"col-status\":{\"label\":\"Status Code\",\"dataType\":\"number\",\"operationType\":\"terms\",\"scale\":\"ordinal\",\"sourceField\":\"status_code\",\"isBucketed\":true,\"params\":{\"size\":10,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col-count\"},\"orderDirection\":\"desc\",\"otherBucket\":true,\"missingBucket\":false,\"parentFormat\":{\"id\":\"terms\"},\"include\":[],\"exclude\":[],\"includeIsRegex\":false,\"excludeIsRegex\":false}},\"col-count\":{\"label\":\"Count\",\"dataType\":\"number\",\"operationType\":\"count\",\"isBucketed\":false,\"scale\":\"ratio\",\"sourceField\":\"___records___\",\"params\":{\"emptyAsNull\":true}}},\"columnOrder\":[\"col-timestamp\",\"col-status\",\"col-count\"],\"incompleteColumns\":{},\"sampling\":1}}}}}},\"enhancements\":{}},\"title\":\"Request Timeline\"},{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":0,\"y\":12,\"w\":16,\"h\":12,\"i\":\"panel-2\"},\"panelIndex\":\"panel-2\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Status Code Distribution\",\"description\":\"Distribution of HTTP status codes\",\"visualizationType\":\"lnsPie\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"web-logs-*\"}],\"state\":{\"visualization\":{\"shape\":\"pie\",\"layers\":[{\"layerId\":\"layer1\",\"primaryGroups\":[\"col-status\"],\"metrics\":[\"col-count\"],\"numberDisplay\":\"percent\",\"categoryDisplay\":\"default\",\"legendDisplay\":\"default\",\"nestedLegend\":false,\"layerType\":\"data\",\"legendPosition\":\"right\",\"percentDecimals\":1,\"emptySizeRatio\":0.3,\"legendMaxLines\":1,\"truncateLegend\":true}]},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col-status\":{\"label\":\"Status Code\",\"dataType\":\"number\",\"operationType\":\"terms\",\"scale\":\"ordinal\",\"sourceField\":\"status_code\",\"isBucketed\":true,\"params\":{\"size\":10,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col-count\"},\"orderDirection\":\"desc\",\"otherBucket\":true,\"missingBucket\":false}},\"col-count\":{\"label\":\"Count\",\"dataType\":\"number\",\"operationType\":\"count\",\"isBucketed\":false,\"scale\":\"ratio\",\"sourceField\":\"___records___\"}},\"columnOrder\":[\"col-status\",\"col-count\"],\"incompleteColumns\":{},\"sampling\":1}}}}}},\"enhancements\":{}},\"title\":\"Status Code Distribution\"},{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":16,\"y\":12,\"w\":16,\"h\":12,\"i\":\"panel-3\"},\"panelIndex\":\"panel-3\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Top 10 IPs\",\"description\":\"Top 10 client IP addresses by request count\",\"visualizationType\":\"lnsDatatable\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"web-logs-*\"}],\"state\":{\"visualization\":{\"layerId\":\"layer1\",\"layerType\":\"data\",\"columns\":[{\"columnId\":\"col-ip\",\"alignment\":\"left\"},{\"columnId\":\"col-count\",\"alignment\":\"right\"}],\"paging\":{\"size\":10,\"enabled\":false},\"headerRowHeight\":\"auto\",\"rowHeight\":\"auto\"},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col-ip\":{\"label\":\"Client IP\",\"dataType\":\"string\",\"operationType\":\"terms\",\"scale\":\"ordinal\",\"sourceField\":\"client_ip.keyword\",\"isBucketed\":true,\"params\":{\"size\":10,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col-count\"},\"orderDirection\":\"desc\",\"otherBucket\":false,\"missingBucket\":false}},\"col-count\":{\"label\":\"Requests\",\"dataType\":\"number\",\"operationType\":\"count\",\"isBucketed\":false,\"scale\":\"ratio\",\"sourceField\":\"___records___\"}},\"columnOrder\":[\"col-ip\",\"col-count\"],\"incompleteColumns\":{},\"sampling\":1}}}}}},\"enhancements\":{}},\"title\":\"Top 10 IPs\"},{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":32,\"y\":12,\"w\":16,\"h\":12,\"i\":\"panel-4\"},\"panelIndex\":\"panel-4\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Geographic Distribution\",\"description\":\"Requests by country\",\"visualizationType\":\"lnsXY\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"web-logs-*\"}],\"state\":{\"visualization\":{\"legend\":{\"isVisible\":false},\"valueLabels\":\"hide\",\"fittingFunction\":\"None\",\"axisTitlesVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"tickLabelsVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"labelsOrientation\":{\"x\":-45,\"yLeft\":0,\"yRight\":0},\"gridlinesVisibilitySettings\":{\"x\":true,\"yLeft\":true,\"yRight\":true},\"preferredSeriesType\":\"bar_horizontal\",\"layers\":[{\"layerId\":\"layer1\",\"accessors\":[\"col-count\"],\"position\":\"top\",\"seriesType\":\"bar_horizontal\",\"showGridlines\":false,\"layerType\":\"data\",\"xAccessor\":\"col-country\"}]},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col-country\":{\"label\":\"Country\",\"dataType\":\"string\",\"operationType\":\"terms\",\"scale\":\"ordinal\",\"sourceField\":\"geo.geo.country_name.keyword\",\"isBucketed\":true,\"params\":{\"size\":10,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col-count\"},\"orderDirection\":\"desc\",\"otherBucket\":true,\"missingBucket\":false}},\"col-count\":{\"label\":\"Requests\",\"dataType\":\"number\",\"operationType\":\"count\",\"isBucketed\":false,\"scale\":\"ratio\",\"sourceField\":\"___records___\"}},\"columnOrder\":[\"col-country\",\"col-count\"],\"incompleteColumns\":{},\"sampling\":1}}}}}},\"enhancements\":{}},\"title\":\"Geographic Distribution\"}]","timeRestore":false,"title":"Log Analysis Dashboard","version":1},"coreMigrationVersion":"8.8.0","created_at":"2024-12-21T00:00:00.000Z","id":"log-analysis-dashboard","managed":false,"references":[],"type":"dashboard","typeMigrationVersion":"8.9.0","updated_at":"2024-12-21T00:00:00.000Z","version":"WzEsMV0="}
{"attributes":{"description":"DDoS Detection monitoring","kibanaSavedObjectMeta":{"searchSourceJSON":"{}"},"optionsJSON":"{\"useMargins\":true}","panelsJSON":"[{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":0,\"y\":0,\"w\":24,\"h\":10,\"i\":\"1\"},\"panelIndex\":\"1\",\"embeddableConfig\":{\"attributes\":{\"title\":\"DDoS Detections Over Time\",\"visualizationType\":\"lnsXY\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"ddos-logs\"}],\"state\":{\"visualization\":{\"legend\":{\"isVisible\":true,\"position\":\"right\"},\"valueLabels\":\"hide\",\"preferredSeriesType\":\"bar_stacked\",\"layers\":[{\"layerId\":\"layer1\",\"seriesType\":\"bar_stacked\",\"xAccessor\":\"col1\",\"accessors\":[\"col2\"],\"splitAccessor\":\"col3\",\"layerType\":\"data\"}]},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col1\":{\"label\":\"Timestamp\",\"dataType\":\"date\",\"operationType\":\"date_histogram\",\"sourceField\":\"timestamp\",\"params\":{\"interval\":\"1m\"}},\"col2\":{\"label\":\"Count\",\"dataType\":\"number\",\"operationType\":\"count\",\"sourceField\":\"___records___\",\"isBucketed\":false},\"col3\":{\"label\":\"Status\",\"dataType\":\"string\",\"operationType\":\"terms\",\"sourceField\":\"status\",\"params\":{\"size\":5,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col2\"},\"orderDirection\":\"desc\"}}},\"columnOrder\":[\"col1\",\"col3\",\"col2\"],\"incompleteColumns\":{}}}}}}},\"enhancements\":{}},\"title\":\"DDoS Detections Over Time\"},{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":24,\"y\":0,\"w\":12,\"h\":10,\"i\":\"2\"},\"panelIndex\":\"2\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Detection Status\",\"visualizationType\":\"lnsPie\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"ddos-logs\"}],\"state\":{\"visualization\":{\"shape\":\"pie\",\"layers\":[{\"layerId\":\"layer1\",\"primaryGroups\":[\"col1\"],\"metrics\":[\"col2\"],\"numberDisplay\":\"percent\",\"categoryDisplay\":\"default\",\"legendDisplay\":\"default\",\"layerType\":\"data\"}]},\"query\":{\"query\":\"\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col1\":{\"label\":\"Status\",\"dataType\":\"string\",\"operationType\":\"terms\",\"sourceField\":\"status\",\"params\":{\"size\":5,\"orderBy\":{\"type\":\"column\",\"columnId\":\"col2\"},\"orderDirection\":\"desc\"}},\"col2\":{\"label\":\"Count\",\"dataType\":\"number\",\"operationType\":\"count\",\"sourceField\":\"___records___\",\"isBucketed\":false}},\"columnOrder\":[\"col1\",\"col2\"],\"incompleteColumns\":{}}}}}}},\"enhancements\":{}},\"title\":\"Normal vs DDoS\"},{\"version\":\"8.11.0\",\"type\":\"lens\",\"gridData\":{\"x\":36,\"y\":0,\"w\":12,\"h\":10,\"i\":\"3\"},\"panelIndex\":\"3\",\"embeddableConfig\":{\"attributes\":{\"title\":\"Avg Probability\",\"visualizationType\":\"lnsMetric\",\"type\":\"lens\",\"references\":[{\"type\":\"index-pattern\",\"name\":\"indexpattern-datasource-layer-layer1\",\"id\":\"ddos-logs\"}],\"state\":{\"visualization\":{\"layerId\":\"layer1\",\"layerType\":\"data\",\"metricAccessor\":\"col1\"},\"query\":{\"query\":\"status:DDOS\",\"language\":\"kuery\"},\"filters\":[],\"datasourceStates\":{\"formBased\":{\"layers\":{\"layer1\":{\"columns\":{\"col1\":{\"label\":\"Avg Probability\",\"dataType\":\"number\",\"operationType\":\"average\",\"sourceField\":\"probability\",\"params\":{\"format\":{\"id\":\"percent\",\"params\":{\"decimals\":1}}}}},\"columnOrder\":[\"col1\"],\"incompleteColumns\":{}}}}}}},\"enhancements\":{}},\"title\":\"DDoS Probability\"}]","timeRestore":false,"title":"DDoS Detection Dashboard","version":1},"coreMigrationVersion":"8.8.0","created_at":"2024-12-21T00:00:00.000Z","id":"ddos-detection-dashboard","managed":false,"references":[],"type":"dashboard","typeMigrationVersion":"8.9.0","updated_at":"2024-12-21T00:00:00.000Z","version":"WzIsMV0="}
DASHBOARD_EOF

# Import dashboards
echo "  Importing dashboard objects..."
IMPORT_RESULT=$(curl -sf -X POST "$KIBANA_URL/api/saved_objects/_import?overwrite=true" \
  -H "kbn-xsrf: true" \
  -F "file=@$DASHBOARD_FILE" 2>&1) || true

if echo "$IMPORT_RESULT" | grep -q '"success":true'; then
    echo "Dashboards imported successfully!"
else
    echo "Dashboard import completed (may have warnings)"
    echo "Note: Some visualizations may need manual adjustment for your data"
fi

# Cleanup
rm -f "$DASHBOARD_FILE"

echo ""
echo "========================================"
echo "Import Complete!"
echo "========================================"
echo ""
echo "Access Kibana: $KIBANA_URL"
echo ""
echo "Available Dashboards:"
echo ""
echo "  1. Log Analysis Dashboard"
echo "     - Request Timeline (Line Chart, 1 min interval, split by status_code)"
echo "     - Status Code Distribution (Pie Chart)"
echo "     - Top 10 IPs (Data Table)"
echo "     - Geographic Distribution (Horizontal Bar by Country)"
echo ""
echo "  2. DDoS Detection Dashboard"
echo "     - Detections over time (NORMAL vs DDOS)"
echo "     - Detection status pie chart"
echo "     - Average detection probability"
echo ""
echo "Steps to view:"
echo "  1. Go to $KIBANA_URL"
echo "  2. Navigate to Analytics > Dashboard"
echo "  3. Select a dashboard"
echo ""
