// src/components/ResultDetail.jsx
import { useParams, Link } from "react-router-dom";

export default function ResultDetail() {
    const { id } = useParams();

    const data = {
        name: "홍 민 기",
        birth: "2001. 01. 20",
        gender: "남성",
        testedAt: "2025. 07. 24",
        face: { status: "주의", image: "/images/placeholder_face.png" },
        arm: {
            status: "정상",
            before: "/images/placeholder_arm_before.png",
            after: "/images/placeholder_arm_after.png",
        },
        speech: {
            status: "정상",
            before: "/images/placeholder_wave_before.png",
            after: "/images/placeholder_wave_after.png",
        },
        summary: "",
    };

    const statusColor = (s) => (s === "주의" ? "text-red-500" : "text-blue-600");

    return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-[1px] overflow-y-auto p-6">
        {/* A4 카드: 각진 직사각형 */}
        <div
            className="relative mx-auto bg-white shadow-2xl border border-gray-200 flex flex-col"
            style={{ aspectRatio: "210 / 297", width: "min(92vw, 794px)" }}
        >
            {/* 헤더 */}
            <div className="px-6 sm:px-8 pt-6">
                <h1 className="text-center text-7xl font-extrabold text-blue-600 tracking-wide">F.A.S.T</h1>

                {/* 인적사항 */}
                <div className="mt-10 border border-gray-100 bg-gray-100 rounded-xl overflow-hidden">
                    <div className="grid grid-cols-4 text-center text-[15px] font-semibold text-gray-700">
                        <div className="py-2">이름</div>
                        <div className="py-2">생년월일</div>
                        <div className="py-2">성별</div>
                        <div className="py-2">검사일시</div>
                    </div>
                    <div className="h-px mx-4 bg-gray-300" />
                    <div className="grid grid-cols-4 text-center text-[15px]">
                        <div className="py-2">{data.name}</div>
                        <div className="py-2">{data.birth}</div>
                        <div className="py-2">{data.gender}</div>
                        <div className="py-2">{data.testedAt}</div>
                    </div>
                </div>
            </div>

            {/* 본문: ✅ 내부 스크롤 제거 (overflow-auto 삭제) → 오버레이가 전체 스크롤 담당 */}
            <div className="flex-1 px-6 sm:px-8 pb-6 mt-5">
                {/* Face */}
                <section className="mb-4">
                    <div className="flex items-baseline gap-2">
                        <h2 className="text-[20px] font-semibold text-gray-800">Face 분석 |</h2>
                        <span className={`text-[21px] font-bold ${statusColor(data.face.status)}`}> {data.face.status}</span>
                    </div>

                    {/* 가운데 정렬 */}
                    <div className="mt-0.5 grid grid-cols-[350px_1fr] gap-6 rounded-xl bg-gray-100 p-4">
                        <div className="w-[330px] rounded-xl border border-gray-300 bg-white overflow-hidden flex items-center justify-center">
                            <img
                                src="/facetest.png"
                                alt="Face 분석 이미지"  /* 테스트용 사진이라 연동할 때 수정 */
                                className="w-full h-auto object-contain"
                            />
                        </div>
                        <div className="p-2 text-sm text-black">
                            Face 결과지 내용 입력 예정
                        </div>
                    </div>
                </section>

                {/* Arm */}
                <section className="mb-4">
                    <div className="flex items-baseline gap-2">
                        <h2 className="text-[20px] font-semibold text-gray-800">Arm 분석 |</h2>
                        <span className={`text-[21px] font-bold ${statusColor(data.arm.status)}`}> {data.arm.status}</span>
                    </div>

                    {/* 가운데 정렬 */}
                    <div className="mt-0.5 grid grid-cols-[350px_1fr] gap-6 rounded-xl bg-gray-100 p-4">
                        <div className="w-[330px] rounded-xl border border-gray-300 bg-white overflow-hidden flex items-center justify-center">
                            <img
                                src="/armtest.png"
                                alt="Face 분석 이미지"  /* 테스트용 사진이라 연동할 때 수정 */
                                className="w-full h-auto object-contain"
                            />
                        </div>
                        <div className="p-2 text-sm text-black">
                            Arm 결과지 내용 입력 예정
                        </div>
                    </div>
                </section>

                {/* Speech */}
                <section className="mb-4">
                    <div className="flex items-baseline gap-2">
                        <h2 className="text-[20px] font-semibold text-gray-800">Speech 분석 |</h2>
                        <span className={`text-[21px] font-bold ${statusColor(data.speech.status)}`}> {data.speech.status} </span>
                    </div>

                    {/* 가운데 정렬 */}
                    <div className="mt-0.5 grid grid-cols-[350px_1fr] gap-6 rounded-xl bg-gray-100 p-4">
                        {/* 이미지 2개를 가로로 나란히 */}
                        <div className="flex gap-4">
                            {/* 첫 번째 이미지 */}
                            <div className="w-[170px] aspect-square rounded-xl border border-gray-300 bg-white overflow-hidden flex items-center justify-center">
                                <img
                                    src="/speechtest1.png"
                                    alt="Speech 처음 파장"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                            {/* 두 번째 이미지 */}
                            <div className="w-[170px] aspect-square rounded-xl border border-gray-300 bg-white overflow-hidden flex items-center justify-center">
                                <img
                                    src="/speechtest2.png"
                                    alt="Speech 나중 파장"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>
                        {/* 결과 텍스트 */}
                        <div className="p-2 text-sm text-black">
                            Speech 결과지 내용 입력 예정
                        </div>
                    </div>
                </section>

                {/* 종합분석 */}
                <section className="mt-4">
                    <h2 className="text-[22px] font-bold text-gray-900">종합분석</h2>
                    <div className="mt-0.5 rounded-xl bg-gray-50 p-4">
                        <div className="p-2 text-sm text-gray-700 min-h-[120px]">
                            {data.summary || "종합 분석 내용 입력 예정"}
                        </div>
                    </div>
                </section>

                {/* 닫기 버튼 */}
                <div className="mt-6 flex justify-center">
                    <Link
                        to="/results"
                        className="px-6 py-2 text-[18px] bg-blue-600 text-white rounded-lg shadow hover:brightness-110 active:scale-95 transition"
                    >
                        닫기
                    </Link>
                </div>
            </div>
        </div>
    </div>
);
}
