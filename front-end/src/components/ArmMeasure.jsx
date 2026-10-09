import React, { useEffect, useRef, useState } from "react";
import { postArmPredict } from "@/api/arm";
import { useNavigate } from "react-router-dom";

export default function ArmMeasure() {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const leftBoxRef = useRef(null);
  const rightBoxRef = useRef(null);
  const navigate = useNavigate();
  // MediaPipe Hands 인스턴스
  const handsRef = useRef(null);

  // 자동 시작 제어
  const guideStartRef = useRef(null);   // 박스 안에서 유지 시작 시각
  const startedRef = useRef(false);     // run() 중복 방지
  const cooldownRef = useRef(0);        // 자동 재시작 쿨다운(ms)

  // 미리보기 URL
  const b025UrlRef = useRef(null);
  const b105UrlRef = useRef(null);

  // 상태
  const [log, setLog] = useState([]);
  const [img025, setImg025] = useState(null);
  const [img105, setImg105] = useState(null);
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);

  const pushLog = (m) =>
    setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${m}`]);

  // 비디오 준비 대기
  const waitVideoReady = () =>
    new Promise((resolve) => {
      const v = videoRef.current;
      if (!v) return resolve();
      if (v.readyState >= 2) return resolve(); // HAVE_CURRENT_DATA
      const onReady = () => {
        v.removeEventListener("loadeddata", onReady);
        resolve();
      };
      v.addEventListener("loadeddata", onReady);
    });

  // 프레임 캡처 (미러링 상태로 저장)
  const captureFrame = () => {
    const v = videoRef.current;
    const c = captureCanvasRef.current;
    const w = v?.videoWidth || 1280;
    const h = v?.videoHeight || 720;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1); // 화면과 동일하게 좌우 반전
    ctx.drawImage(v, 0, 0, w, h);
    ctx.restore();
    return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/png"));
  };

  // 측정 루틴 (2.5s → 10.5s)
  const run = async () => {
    if (startedRef.current) return; // 중복 방지
    startedRef.current = true;
    setResp(null);
    setImg025(null);
    setImg105(null);

    // 기존 미리보기 URL 정리
    if (b025UrlRef.current) URL.revokeObjectURL(b025UrlRef.current);
    if (b105UrlRef.current) URL.revokeObjectURL(b105UrlRef.current);
    b025UrlRef.current = null;
    b105UrlRef.current = null;

    setLoading(true);
    pushLog("측정 시작");

    try {
      pushLog("2.5초 대기…");
      await new Promise((r) => setTimeout(r, 2500));
      const b025 = await captureFrame();
      b025UrlRef.current = URL.createObjectURL(b025);
      setImg025(b025UrlRef.current);
      pushLog("2.5s 캡처 완료");

      pushLog("10.5초(추가 8초) 대기…");
      await new Promise((r) => setTimeout(r, 8000));
      const b105 = await captureFrame();
      b105UrlRef.current = URL.createObjectURL(b105);
      setImg105(b105UrlRef.current);
      pushLog("10.5s 캡처 완료");

      const data = await postArmPredict(b025, b105);
      setResp(data);
      pushLog("서버 응답 수신");
      navigate("/test/speech", { replace: true });
    } catch (e) {
      const msg = e?.message || String(e);
      pushLog("오류: " + msg);
      alert(msg);
    } finally {
      setLoading(false);
      cooldownRef.current = Date.now() + 3000; // 3초 재시작 쿨다운
      startedRef.current = false;
      guideStartRef.current = null;
    }
  };

  // MediaPipe Hands 결과 처리 → “좌/우 박스 모두 3초 유지 시 run()”
  const onResults = (results) => {
    const video = videoRef.current;
    const leftBox = leftBoxRef.current;
    const rightBox = rightBoxRef.current;
    if (!video || !leftBox || !rightBox) return;

    const videoRect = video.getBoundingClientRect();
    const leftRect = leftBox.getBoundingClientRect();
    const rightRect = rightBox.getBoundingClientRect();

    let inLeft = false;
    let inRight = false;

    const lmSets = results?.multiHandLandmarks || [];
    const handed = results?.multiHandedness || [];

    lmSets.forEach((lm, idx) => {
      const tip = lm[12]; // 가운데 손가락 tip
      const raw = handed?.[idx];
      const label =
        raw?.label ??
        raw?.classification?.[0]?.label ??
        (tip?.x < 0.5 ? "Left" : "Right"); // 폴백

      // selfieMode: true이므로 결과 좌표는 화면과 동일한 방향
      const tipX = videoRect.left + tip.x * videoRect.width;
      const tipY = videoRect.top + tip.y * videoRect.height;

      if (
        label === "Left" &&
        tipX >= leftRect.left &&
        tipX <= leftRect.right &&
        tipY >= leftRect.top &&
        tipY <= leftRect.bottom
      ) {
        inLeft = true;
      }
      if (
        label === "Right" &&
        tipX >= rightRect.left &&
        tipX <= rightRect.right &&
        tipY >= rightRect.top &&
        tipY <= rightRect.bottom
      ) {
        inRight = true;
      }
    });

    const bothInside = inLeft && inRight;

    if (!startedRef.current && bothInside && Date.now() > cooldownRef.current) {
      if (!guideStartRef.current) {
        guideStartRef.current = performance.now();
      } else if (performance.now() - guideStartRef.current > 3000) {
        pushLog("3초 유지 완료 — 자동 측정 시작");
        run();
      }
    } else {
      guideStartRef.current = null;
    }
  };

  // 초기화: 카메라 + MediaPipe Hands (v0.4, CDN 스크립트 주입)
  useEffect(() => {
    let running = true;

    // CDN 스크립트 로더 (중복 삽입 방지)
    const loadScript = (src) =>
      new Promise((resolve, reject) => {
        if ([...document.scripts].some((s) => s.src === src)) return resolve();
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`Failed to load: ${src}`));
        document.head.appendChild(s);
      });

    (async () => {
      // 1) 카메라
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
      await waitVideoReady();

      // 2) MediaPipe Hands v0.4 + camera_utils 로드
      const HANDS_VER = "0.4";
      await loadScript(`https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VER}/hands.js`);
      try {
        await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js");
      } catch {
        await loadScript("https://unpkg.com/@mediapipe/camera_utils/camera_utils.js");
      }
      // (선택) 드로잉이 필요하면 다음 줄도:
      // await loadScript(`https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@${HANDS_VER}/drawing_utils.js`);

      const HandsCtor = window.Hands;
      const CameraCtor = window.Camera;
      if (!HandsCtor || !CameraCtor) {
        throw new Error("MediaPipe Hands 또는 Camera 유틸을 불러오지 못했습니다.");
      }

      // 3) Hands 인스턴스 생성
      const hands = new HandsCtor({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${HANDS_VER}/${file}`,
      });
      hands.setOptions({
        selfieMode: true,        // 좌우 화면과 결과 좌표계 일치
        maxNumHands: 2,
        modelComplexity: 1,      // 0/1
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      hands.onResults(onResults);
      handsRef.current = hands;

      // 4) Camera 유틸로 프레임 공급
      const camera = new CameraCtor(videoRef.current, {
        onFrame: async () => {
          if (!running) return;
          await hands.send({ image: videoRef.current });
        },
        width: 1280,
        height: 720,
      });
      camera.start();

      pushLog("핸드 추적 시작 (MediaPipe Hands v0.4 + Camera)");
    })().catch((err) => {
      const msg = err?.message || String(err);
      pushLog("초기화 오류: " + msg);
      alert(msg);
    });

    return () => {
      running = false;
      try {
        handsRef.current?.close?.();
      } catch {}
      const tracks = videoRef.current?.srcObject?.getTracks?.();
      tracks?.forEach((t) => t.stop());
      if (b025UrlRef.current) URL.revokeObjectURL(b025UrlRef.current);
      if (b105UrlRef.current) URL.revokeObjectURL(b105UrlRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // === JSX ===
  return (
    <div className="w-screen h-screen flex items-center justify-center bg-white relative overflow-hidden">
      <div className="w-[133vh] h-[133vh] rounded-full border-[7vw] border-[#f6f6f6] shadow-xl overflow-hidden flex items-center justify-center z-0 relative">
        {/* 카메라와 가이드 박스 */}
        <div
          style={{
            position: "relative",
            width: "100%",
            maxWidth: 850,
            aspectRatio: "16/9",
            borderRadius: 12,
            overflow: "hidden",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scaleX(-1)", // 화면만 미러링 (selfieMode는 결과 좌표에 적용)
            }}
          />
          {/* 가이드 박스: 좌/우 */}
          <div
            id="left-box"
            ref={leftBoxRef}
            style={{
              position: "absolute",
              left: "8%",
              top: "20%",
              width: "26%",
              height: "60%",
              border: "3px solid #4ade80",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(74,222,128,0.5) inset",
            }}
          />
          <div
            id="right-box"
            ref={rightBoxRef}
            style={{
              position: "absolute",
              right: "8%",
              top: "20%",
              width: "26%",
              height: "60%",
              border: "3px solid #60a5fa",
              borderRadius: 12,
              boxShadow: "0 0 12px rgba(96,165,250,0.5) inset",
            }}
          />
        </div>
      </div>

      {/* 우측 응답/미리보기 영역 (수동 버튼 제거) */}
      <div
        style={{
          position: "absolute",
          right: 32,
          top: 32,
          width: 400,
          background: "#121833",
          border: "1px solid #2a3566",
          borderRadius: 12,
          padding: 12,
          zIndex: 10,
        }}
      >
        <h2 style={{ marginTop: 0, color: "#cfe3ff" }}>응답</h2>
        <pre
          style={{
            margin: 0,
            whiteSpace: "pre-wrap",
            fontFamily: "ui-monospace,monospace",
            color: "#cfe3ff",
          }}
        >
          {resp ? JSON.stringify(resp, null, 2) : "{ 아직 없음 }"}
        </pre>

        {(img025 || img105) && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 12,
              marginTop: 12,
            }}
          >
            <figure style={{ margin: 0 }}>
              <figcaption style={{ color: "#cfe3ff" }}>2.5s 프레임</figcaption>
              {img025 && (
                <img
                  src={img025}
                  alt="2.5s"
                  style={{ width: "100%", borderRadius: 8 }}
                />
              )}
            </figure>
            <figure style={{ margin: 0 }}>
              <figcaption style={{ color: "#cfe3ff" }}>10.5s 프레임</figcaption>
              {img105 && (
                <img
                  src={img105}
                  alt="10.5s"
                  style={{ width: "100%", borderRadius: 8 }}
                />
              )}
            </figure>
          </div>
        )}

        <div
          style={{
            marginTop: 10,
            fontFamily: "ui-monospace,monospace",
            fontSize: 13,
            color: "#a9b4d4",
            whiteSpace: "pre-wrap",
          }}
        >
          {log.join("\n") || "{ 로그 없음 }"}
        </div>
      </div>

      <canvas ref={captureCanvasRef} style={{ display: "none" }} />
    </div>
  );
}
