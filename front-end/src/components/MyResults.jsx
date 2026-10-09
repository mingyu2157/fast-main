import { useMemo, useState } from "react";
import { Link as RouterLink, useNavigate, useLocation } from "react-router-dom";
import { Link as LinkIcon, Home, Menu } from "lucide-react";
import TopRightMenu from "./TopRightMenu"; // 이미 있다면 그대로 사용

// 상태 뱃지
function StatusBadge({ value }) {
    const isWarn = value === "경고";
    return (
        <span className={`text-2xl ${isWarn ? "text-red-500 font-bold" : "text-gray-800"}`}>
      {value}
    </span>
    );
}

export default function MyResults() {
    const navigate = useNavigate();
    const location = useLocation(); // ✅ 모달 라우트용 배경 전달

    // TODO: 이후 백엔드 연동 시 API 호출로 대체
    const allData = useMemo(
        () => [
            { id: 6, date: "2025.09.01", face: "경고", arm: "정상", speech: "정상"},
            { id: 5, date: "2025.07.21", face: "정상", arm: "경고", speech: "경고" },
            { id: 4, date: "2025.07.19", face: "정상", arm: "정상", speech: "정상" },
            { id: 3, date: "2025.07.15", face: "정상", arm: "경고", speech: "정상" },
            { id: 2, date: "2025.07.10", face: "정상", arm: "정상", speech: "정상" },
            { id: 1, date: "2025.07.01", face: "정상", arm: "정상", speech: "정상" },
            // 필요하면 더미 더 추가
        ],
        []
    );

    const pageSize = 5;
    const [page, setPage] = useState(1);
    const totalPages = Math.max(1, Math.ceil(allData.length / pageSize));
    const pageData = useMemo(() => {
        const start = (page - 1) * pageSize;
        return allData.slice(start, start + pageSize);
    }, [allData, page]);

    const openDetail = (row) => {
        // ✅ 배경 location 함께 전달 → /results/:id 를 모달로 띄우고 뒤에 MyResults 유지
        navigate(`/results/${row.id}`, { state: { background: location } });
    };

    return (
        <div className="relative min-h-screen bg-white px-6 py-8">
            {/* 상단 오른쪽 고정 메뉴 (이미 프로젝트에 있다면 유지) */}
            <TopRightMenu showLoginButton={false} showHomeButton={true} />

            <div className="text-center mt-4">
                {/* 제목 */}
                <h1 className="text-7xl font-bold text-blue-600 mt-16 mb-10">
                    My 검사결과
                </h1>
                <p className="inline-block bg-gray-100 px-12 py-1.5 text-[1.6rem] rounded-full text-black">
                    F.A.S.T 검사 결과를 확인할 수 있습니다.
                </p>

                {/* 표 */}
                <div className="mx-36 mt-20 overflow-hidden rounded-2xl shadow-[0_10px_30px_rgba(0,0,0,0.08)]">
                    <table className="w-full table-fixed">
                        <thead className="bg-white">
                        <tr className="border-b-2">
                            <th className="w-[8%] py-5 text-2xl font-semibold text-gray-700">번호</th>
                            <th className="w-[18%] py-5 text-2xl font-semibold text-gray-700">날짜</th>
                            <th className="w-[18%] py-5 text-2xl font-semibold text-gray-700">Face 결과</th>
                            <th className="w-[18%] py-5 text-2xl font-semibold text-gray-700">Arm 결과</th>
                            <th className="w-[18%] py-5 text-2xl font-semibold text-gray-700">Speech 결과</th>
                            <th className="w-[20%] py-5 text-2xl font-semibold text-gray-700">결과지 링크</th>
                        </tr>
                        </thead>
                        <tbody>
                        {pageData.map((row) => (
                            <tr key={row.id} className="border-b last:border-b-0">
                                <td className="py-4 text-center text-2xl text-gray-900">{row.id}</td>
                                <td className="py-4 text-center text-2xl text-gray-900">{row.date}</td>
                                <td className="py-4 text-center"><StatusBadge value={row.face} /></td>
                                <td className="py-4 text-center"><StatusBadge value={row.arm} /></td>
                                <td className="py-4 text-center"><StatusBadge value={row.speech} /></td>
                                <td className="py-4">
                                    <div className="flex justify-center">
                                        <button
                                            onClick={() => openDetail(row)}
                                            className="p-2 rounded-xl hover:bg-gray-100 active:scale-95 transition"
                                            aria-label="결과지 열기"
                                            title="결과지 열기"
                                        >
                                            <LinkIcon className="w-8 h-8 text-gray-600" />
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {pageData.length === 0 && (
                            <tr>
                                <td colSpan={6} className="py-16 text-center text-gray-500 text-xl">
                                    데이터가 없습니다.
                                </td>
                            </tr>
                        )}
                        </tbody>
                    </table>
                </div>

                {/* 페이지네이션 */}
                <div className="mt-10 flex items-center justify-center gap-4">
                    {Array.from({ length: totalPages }).map((_, i) => {
                        const n = i + 1;
                        const isActive = n === page;
                        return (
                            <button
                                key={n}
                                onClick={() => setPage(n)}
                                className={`w-12 h-12 rounded-full text-2xl transition
                ${isActive ? "bg-blue-600 text-white shadow-lg scale-105" : "bg-white text-gray-800 border"}
              `}
                            >
                                {n}
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
