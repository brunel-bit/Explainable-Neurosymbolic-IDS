from src.integration.comat_adapter import shap_json_to_comat

output = shap_json_to_comat(
    "outputs/shap/instance_0.json",
    "outputs/comat/input_case_0.json",
)

print(output)