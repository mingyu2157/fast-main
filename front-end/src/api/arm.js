import http from "@/lib/http"; // baseURL: "", withCredentials: true

export async function postArmPredict(startBlob, endBlob, { useMeasure = false } = {}) {
  const fd = new FormData();
  fd.append("start_file", startBlob, "t025.png");
  fd.append("end_file", endBlob, "t105.png");
  const path = useMeasure ? "/api/v1/arm/measure" : "/api/v1/arm/predict";
  const { data } = await http.post(path, fd, { headers: { "Content-Type": "multipart/form-data" } });
  return data;
}

export async function postArmPredictLandmarks(payload) {
  const { data } = await http.post("/api/v1/arm/predict_landmarks", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
}