"""데이터 만들기 단계를 순서대로 한 번에 실행한다. 한 단계라도 실패하면 바로 멈춘다.

실행:
  python run_all.py                 # 기본: 이미 받아 둔 데이터로 계산·분석·검증만 (API 키 불필요, 안전)
  python run_all.py --with-collect  # DART에서 새로 수집하는 단계(collect_*.py)도 포함 (API 키 필요)
  python run_all.py --only metrics  # 한 단계만 (이름은 아래 STEPS 참고)

왜 필요한가: 순서를 손으로 따르다 한 단계를 빼먹거나, 앞 단계가 실패했는데 다음 단계가 낡은 CSV를 읽는 일을 막는다.
수동 입력(rnd_manual.csv, customer_concentration.csv)은 사람이 해야 하므로 여기서 하지 않는다.
"""
import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

# (이름, 파일, DART API를 부르는 단계인가)  -- CLAUDE.md의 실행 순서와 같다
STEPS = [
    ("collect_data", "collect_data.py", True),
    ("metrics", "metrics.py", False),
    ("analyze_capex", "analyze_capex.py", False),
    ("collect_quarterly", "collect_quarterly.py", True),
    ("analyze_quarterly", "analyze_quarterly.py", False),
    ("collect_contracts", "collect_contracts.py", True),
    ("analyze_exposure", "analyze_exposure.py", False),
    ("source_links", "collect_source_links.py", False),   # 캐시만 읽는다
    ("validate", "validate.py", False),
]


def main():
    ap = argparse.ArgumentParser(description="데이터 파이프라인 전체 실행")
    ap.add_argument("--with-collect", action="store_true", help="DART에서 새로 수집하는 단계도 실행")
    ap.add_argument("--only", choices=[s[0] for s in STEPS], help="이 단계 하나만 실행")
    args = ap.parse_args()

    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    else:
        steps = [s for s in STEPS if args.with_collect or not s[2]]

    skipped = [s[0] for s in STEPS if s not in steps]
    if skipped and not args.only:
        print(f"(건너뜀: {', '.join(skipped)} -- DART 수집은 --with-collect 로 실행)\n")

    for i, (name, script, _) in enumerate(steps, 1):
        print(f"===== [{i}/{len(steps)}] {name} ({script}) =====")
        start = time.time()
        # sys.executable: 지금 이 스크립트를 실행한 같은 파이썬(가상환경)으로 다음 단계도 실행한다
        result = subprocess.run([sys.executable, os.path.join(BASE, script)], cwd=BASE)
        if result.returncode != 0:
            print(f"\n[중단] {name} 단계가 실패했습니다(종료 코드 {result.returncode}). 뒤 단계는 실행하지 않았습니다.")
            return result.returncode
        print(f"----- {name} 완료 ({time.time() - start:.1f}초)\n")

    print("모든 단계가 끝났습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
