# Claims Data Schema Contract

Every machine learning layer and orchestration step in this MVP enforces a strict data contract to maintain high integration fidelity. 

## 📋 VLM Analysis Output Schema (JSON)
The local VLM is prompted to strictly return a JSON object conforming to the following Pydantic schema contract:

```json
{
  "claim_id": "string",
  "overall_summary": "string describing visual findings",
  "overall_severity": "Pristine | Minor | Moderate | Severe",
  "damages": [
    {
      "part": "string (affected part, e.g. front bumper)",
      "damage_type": "string (e.g. dent, scratch, cracked)",
      "severity": "Pristine | Minor | Moderate | Severe",
      "supporting_images": ["string filename"],
      "confidence": 0.00,
      "reasoning": "string visual justification"
    }
  ],
  "view_consistency_notes": "string explaining how multi-view images correlate"
}
```

## 💰 Integrated Estimation Cost Schema
Once the `VehicleCostEstimator` processes the VLM claim, the schema is augmented with standard Audatex/Mitchell style line item sheets:

```json
{
  "claim_id": "string",
  "overall_summary": "string",
  "overall_severity": "Pristine | Minor | Moderate | Severe",
  "damages": [ ... ],
  "view_consistency_notes": "string",
  "cost_estimation": {
    "vehicle_classification": "Luxury | Import | Economy",
    "hourly_rates": {
      "body_labor_rate_usd": 120.0,
      "paint_labor_rate_usd": 125.0,
      "paint_material_rate_usd": 45.0
    },
    "summary_totals": {
      "total_parts_cost": 0.00,
      "total_body_labor_hours": 0.0,
      "total_body_labor_cost": 0.00,
      "total_paint_labor_hours": 0.0,
      "total_paint_labor_cost": 0.00,
      "total_paint_material_cost": 0.00,
      "grand_total_estimate": 0.00
    },
    "line_items": [
      {
        "part": "string",
        "severity": "string",
        "decision": "Repair | Replace | None",
        "part_type": "OEM | Aftermarket | N/A",
        "part_cost": 0.00,
        "labor_hours_body": 0.0,
        "labor_hours_paint": 0.0,
        "cost_labor_body": 0.00,
        "cost_labor_paint": 0.00,
        "cost_paint_material": 0.00,
        "total_item_cost": 0.00,
        "justification": "string"
      }
    ]
  }
}
```
