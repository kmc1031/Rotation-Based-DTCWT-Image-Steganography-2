import numpy as np
from dtcwt import Transform2d
from dtcwt.numpy import Pyramid
from PIL import Image
import argparse

# =========================
# 유틸리티
# =========================


def text_to_bits(text: str) -> str:
    """텍스트를 '0'/'1' 비트 문자열로 변환"""
    return "".join(format(ord(c), "08b") for c in text)


def bits_to_text(bits: str) -> str:
    """비트 문자열을 텍스트로 변환 (8비트씩)"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def sgn(x: float) -> int:
    """부호 함수: 0을 포함해 비음이 아니면 +1, 음수면 -1"""
    return 1 if x >= 0 else -1


def ensure_signed_magnitude(
    current_real: float, target_sign: int, min_mag: float
) -> float:
    """목표 부호와 최소 크기를 만족하도록 실수부를 조정"""
    mag = max(abs(current_real), float(min_mag))
    return mag if target_sign >= 0 else -mag


def get_orientation_pairs():
    """
    DTCWT 6방향 밴드의 대칭쌍 인덱스.
    일반적으로 dtcwt는 [ +15, +45, +75, -75, -45, -15 ] 순서를 사용합니다.
    이에 대응하는 대칭쌍은 (0,5), (1,4), (2,3)로 가정합니다.
    """
    return [(0, 5), (1, 4), (2, 3)]


def generate_positions(
    highpasses, levels: int, stride: int, key: int, min_coeff_mag: float = 0.0
):
    """
    삽입/추출 좌표 시퀀스를 생성(비밀키 기반 난수 순열).
    반환: [(level_idx, pair_idx, y, x), ...]
    - level_idx: 0..levels-1 중에서 사용
    - pair_idx: 대칭쌍 인덱스(0..2)
    - min_coeff_mag: 계수 필터링 임계값 (양쪽 계수 모두 이 값 이상이어야 사용)
    """
    rng = np.random.RandomState(int(key))
    orient_pairs = get_orientation_pairs()

    positions = []
    # 실용적으로 상위 2레벨 정도가 강인성이 좋으므로 levels를 그대로 사용하되
    # 사용자가 levels를 작게 주면 그 범위 내에서 전부 사용
    use_level_indices = list(range(min(len(highpasses), int(levels))))

    for l in use_level_indices:
        band = highpasses[l]  # shape: (H, W, 6), complex
        H, W, _ = band.shape
        for pair_idx, (o1, o2) in enumerate(orient_pairs):
            # stride 간격으로 좌표 샘플링하여 용량/품질 균형
            ys = range(0, H, stride)
            xs = range(0, W, stride)
            for y in ys:
                for x in xs:
                    # 계수 필터링: 양쪽 계수가 모두 충분히 커야만 사용
                    # min_coeff_mag=0이면 필터링 비활성화
                    if min_coeff_mag > 0:
                        c1 = band[y, x, o1]
                        c2 = band[y, x, o2]
                        r1 = abs(float(np.real(c1)))
                        r2 = abs(float(np.real(c2)))

                        # 두 계수 모두 임계값 이상이어야 안정적
                        if r1 >= min_coeff_mag and r2 >= min_coeff_mag:
                            positions.append((l, pair_idx, y, x))
                    else:
                        # 필터링 없음: 모든 위치 사용
                        positions.append((l, pair_idx, y, x))

    # 비밀키로 결정되는 순열
    rng.shuffle(positions)
    return positions


def relation_from_pair(band, y: int, x: int, o1: int, o2: int) -> (int, float):
    """
    대칭쌍 두 계수의 '관계 부호'와 신뢰도(가중치)를 계산.
    - r = sgn(Re(c_o1)) * sgn(Re(c_o2))  # 각 부호의 곱
    - w = min(|Re(c_o1)|, |Re(c_o2)|)
    """
    c1 = band[y, x, o1]
    c2 = band[y, x, o2]
    r1 = float(np.real(c1))
    r2 = float(np.real(c2))
    # 관계 부호는 각 부호의 곱
    r = sgn(r1) * sgn(r2)
    w = min(abs(r1), abs(r2))
    return r, w, r1, r2, c1, c2


# =========================
# 핵심: 삽입 / 추출
# =========================


def embed_message(
    image_path: str,
    message: str,
    output_path: str,
    levels: int = 3,
    strength: float = 30.0,
    redundancy: int = 8,
    key: int = 12345,
    stride: int = 2,
    min_coeff_mag: float = 0.0,
    verify: bool = False,
):
    """
    DTCWT 기반 스테가노그래피: '대칭 방향 결합 + 관계 부호 + 다수결' 삽입

    Parameters:
    - image_path: 입력 그레이스케일 이미지 경로
    - message: 삽입할 문자열
    - output_path: 스테고 이미지 저장 경로
    - levels: DTCWT 분해 레벨 (기본 3)
    - strength: 최소 실수부 크기 보장값(강도, 기본 30.0). 역변환 노이즈 내성
    - redundancy: 비트당 중복 삽입 횟수 (기본 8)
    - key: 좌표 난수 순열 시드(송수신 동일해야 함)
    - stride: 좌표 샘플링 간격(용량/왜곡/속도 조절)
    """
    # 1) 이미지 로드 (Grayscale)
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float64)

    # 2) 메시지 비트열 구성 (길이 32비트 + 본문 + 종료마커)
    message_with_delimiter = message + "###END###"
    bits_body = text_to_bits(message_with_delimiter)
    length_bits = format(len(bits_body), "032b")
    total_bits = length_bits + bits_body
    total_bits_len = len(total_bits)

    # 3) DTCWT 변환
    transform = Transform2d()
    coeffs = transform.forward(gray, nlevels=levels)
    lowpass = coeffs.lowpass.copy()
    highpass = [band.copy() for band in coeffs.highpasses]  # list of arrays (complex)

    # 4) 좌표 시퀀스 생성(비밀키 기반, 작은 계수 필터링). 용량 확인
    positions = generate_positions(
        highpass, levels=levels, stride=stride, key=key, min_coeff_mag=min_coeff_mag
    )
    orient_pairs = get_orientation_pairs()

    needed_positions = redundancy * total_bits_len
    if needed_positions > len(positions):
        # 최대 수용 가능한 비트수와 문자수 안내
        max_bits = len(positions) // redundancy
        max_payload_bits = max(0, max_bits - 32)  # 32비트 길이 헤더 제외
        max_chars = max_payload_bits // 8
        raise ValueError(
            f"용량 부족: 필요 {needed_positions} 위치, 가용 {len(positions)} 위치. "
            f"현재 파라미터(stride/redundancy/levels)를 조정하세요. "
            f"추정 최대 메시지 길이: 약 {max_chars} 문자"
        )

    # 5) 각 비트를 redundancy만큼 분산 삽입
    pos_idx = 0
    for bit_idx, bit in enumerate(total_bits):
        r_target = +1 if bit == "0" else -1

        for _ in range(redundancy):
            l, pair_idx, y, x = positions[pos_idx]
            pos_idx += 1

            o1, o2 = orient_pairs[pair_idx]
            band = highpass[l]  # shape: (H, W, 6), complex

            # 현재 관계 부호와 신뢰도 계산
            r_cur, w, r1, r2, c1, c2 = relation_from_pair(band, y, x, o1, o2)

            # 목표 r에 맞게 최소 변경
            # r = sgn(r1) * sgn(r2)이므로, r_target에 맞게 부호 조정
            if r_cur != r_target:
                # 더 약한 쪽(실수부 절대값 작은 계수)의 부호만 바꿔 비용 최소화
                if abs(r1) <= abs(r2):
                    # r1의 부호를 바꿈: new_sgn(r1) = r_target * sgn(r2)
                    new_s1 = r_target * sgn(r2)
                    new_r1 = ensure_signed_magnitude(r1, new_s1, strength)
                    # r2는 부호 유지, 강도만 보장
                    if abs(r2) >= strength:
                        new_r2 = r2  # 이미 충분히 크면 그대로 유지
                    else:
                        new_r2 = ensure_signed_magnitude(r2, sgn(r2), strength)
                else:
                    # r2의 부호를 바꿈: new_sgn(r2) = r_target * sgn(r1)
                    new_s2 = r_target * sgn(r1)
                    new_r2 = ensure_signed_magnitude(r2, new_s2, strength)
                    # r1은 부호 유지, 강도만 보장
                    if abs(r1) >= strength:
                        new_r1 = r1  # 이미 충분히 크면 그대로 유지
                    else:
                        new_r1 = ensure_signed_magnitude(r1, sgn(r1), strength)
            else:
                # 이미 일치: 원본 값 유지 (왜곡 최소화)
                # 다만, 너무 작으면 노이즈에 취약하므로 최소 크기 보장
                if abs(r1) >= strength * 0.5:
                    new_r1 = r1
                else:
                    new_r1 = ensure_signed_magnitude(r1, sgn(r1), strength)
                if abs(r2) >= strength * 0.5:
                    new_r2 = r2
                else:
                    new_r2 = ensure_signed_magnitude(r2, sgn(r2), strength)

            # 밴드 갱신(실수부만 갱신, 허수부는 보존)
            band[y, x, o1] = complex(new_r1, float(np.imag(c1)))
            band[y, x, o2] = complex(new_r2, float(np.imag(c2)))

            # 검증: 삽입 후 관계 부호가 올바른지 확인
            verify_r = sgn(new_r1) * sgn(new_r2)
            if verify_r != r_target:
                # 디버깅용 경고 (실제로는 발생하지 않아야 함)
                pass  # print(f"[WARN] 삽입 검증 실패: bit={bit}, r_target={r_target}, verify_r={verify_r}")

    # 6) 역변환 및 저장
    stego = transform.inverse(Pyramid(lowpass, highpass))
    stego = stego[: gray.shape[0], : gray.shape[1]]
    stego = np.clip(stego, 0, 255).astype(np.uint8)

    Image.fromarray(stego, "L").save(output_path, "PNG")

    print(f"[OK] 메시지 삽입 완료 → {output_path}")
    print(f"- 총 삽입 비트: {total_bits_len} (길이 헤더 32비트 포함)")
    print(f"- redundancy: {redundancy}, stride: {stride}, levels: {levels}")
    print(f"- 사용 위치: {needed_positions} / 가용 {len(positions)}")

    # 선택적 검증
    if verify:
        # 디버그: 삽입 직후 재추출하여 검증
        coeffs_verify = transform.forward(stego.astype(np.float64), nlevels=levels)
        highpass_verify = [band.copy() for band in coeffs_verify.highpasses]
        positions_verify = generate_positions(
            highpass_verify,
            levels=levels,
            stride=stride,
            key=key,
            min_coeff_mag=min_coeff_mag,
        )

        # 전체 메시지 검증 (처음 100비트만 또는 전체)
        verify_limit = min(100, len(total_bits))  # 긴 메시지는 처음 100비트만 검증
        verify_bits = []
        pos_idx_verify = 0
        failed_positions = []
        for hb in range(verify_limit):
            votes = 0.0
            vote_details = []
            for rep in range(redundancy):
                l, pair_idx, y, x = positions_verify[pos_idx_verify]
                pos_idx_verify += 1
                o1, o2 = orient_pairs[pair_idx]
                band = highpass_verify[l]
                r_cur, w, r1_v, r2_v, _, _ = relation_from_pair(band, y, x, o1, o2)
                vote_val = w * (1 if r_cur >= 0 else -1)
                votes += vote_val
                vote_details.append((l, pair_idx, y, x, r1_v, r2_v, r_cur, w, vote_val))
            bit = "0" if votes >= 0 else "1"
            verify_bits.append(bit)

            # 불일치 검사
            if bit != total_bits[hb]:
                failed_positions.append((hb, total_bits[hb], bit, votes, vote_details))

        original_bits = total_bits[:verify_limit]
        matches = sum(
            1 for i in range(len(verify_bits)) if verify_bits[i] == original_bits[i]
        )

        print(
            f"[DEBUG] 삽입 후 검증: {matches}/{len(verify_bits)} 비트 일치 (처음 {verify_limit}비트)"
        )
        if matches < len(verify_bits):
            print(f"[DEBUG] 실패 비트 수: {len(failed_positions)}")
            if len(failed_positions) <= 3:
                # 실패가 적으면 상세 출력
                print(f"[DEBUG] 원본: {original_bits}")
                print(f"[DEBUG] 추출: {''.join(verify_bits)}")
                print(f"[DEBUG] 실패한 비트 위치:")
                for bit_idx, orig, extr, votes, details in failed_positions:
                    print(
                        f"  비트 {bit_idx}: 원본={orig}, 추출={extr}, votes={votes:.2f}"
                    )
                    for l, pair_idx, y, x, r1, r2, r_cur, w, vote_val in details[
                        :2
                    ]:  # 처음 2개만 출력
                        print(
                            f"    L{l} pair{pair_idx} ({y},{x}): r1={r1:.4f}, r2={r2:.4f}, r_cur={r_cur}, w={w:.4f}, vote={vote_val:.4f}"
                        )


def extract_message(
    image_path: str,
    levels: int = 3,
    redundancy: int = 8,
    key: int = 12345,
    stride: int = 2,
    min_coeff_mag: float = 0.0,
    debug: bool = False,
):
    """
    DTCWT 기반 스테가노그래피: '대칭 방향 결합 + 관계 부호 + 다수결' 추출
    삽입 시와 동일한 levels/redundancy/key/stride를 사용해야 함.
    """
    # 1) 이미지 로드
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float64)

    # 2) 변환
    transform = Transform2d()
    coeffs = transform.forward(gray, nlevels=levels)
    highpass = [band.copy() for band in coeffs.highpasses]
    positions = generate_positions(
        highpass, levels=levels, stride=stride, key=key, min_coeff_mag=min_coeff_mag
    )
    orient_pairs = get_orientation_pairs()

    # 3) 길이 헤더(32비트) 복호
    header_bits = []
    pos_idx = 0
    for hb in range(32):
        votes = 0.0
        for _ in range(redundancy):
            if pos_idx >= len(positions):
                raise ValueError(
                    "용량 부족: 길이 헤더를 복호하기 전에 위치가 소진되었습니다."
                )
            l, pair_idx, y, x = positions[pos_idx]
            pos_idx += 1

            o1, o2 = orient_pairs[pair_idx]
            band = highpass[l]
            r_cur, w, r1, r2, _, _ = relation_from_pair(band, y, x, o1, o2)

            # 가중 다수결(신뢰도 w = min(|r1|,|r2|))
            votes += w * (1 if r_cur >= 0 else -1)

        bit = "0" if votes >= 0 else "1"
        header_bits.append(bit)

    length_bits = "".join(header_bits)
    msg_length = int(length_bits, 2)

    if debug:
        print(f"[DBG] length_bits (32): {length_bits}")
        print(f"[DBG] message length (bits): {msg_length}")

    # 4) 본문 비트 복호
    bits = []
    needed_positions = pos_idx + redundancy * msg_length
    if needed_positions > len(positions):
        # 공격 등으로 길이 해석이 과대 추정된 경우 방어적으로 잘라냄
        max_bits = (len(positions) - pos_idx) // redundancy
        if max_bits < 0:
            max_bits = 0
        if debug:
            print(f"[WARN] 용량 초과. 복호 가능한 최대 비트로 절삭: {max_bits}")
        msg_length = max_bits

    for bi in range(msg_length):
        votes = 0.0
        for _ in range(redundancy):
            l, pair_idx, y, x = positions[pos_idx]
            pos_idx += 1

            o1, o2 = orient_pairs[pair_idx]
            band = highpass[l]
            r_cur, w, r1, r2, _, _ = relation_from_pair(band, y, x, o1, o2)
            votes += w * (1 if r_cur >= 0 else -1)

        bit = "0" if votes >= 0 else "1"
        bits.append(bit)

    bitstr = "".join(bits)
    msg = bits_to_text(bitstr)

    # 5) 종료마커로 잘라내기(안전장치)
    endpos = msg.find("###END###")
    if endpos != -1:
        msg = msg[:endpos]

    return msg


# =========================
# CLI
# =========================


def main():
    parser = argparse.ArgumentParser(
        description="DTCWT 스테가노그래피 (대칭방향+관계부호+다수결)"
    )
    parser.add_argument(
        "mode", choices=["embed", "extract"], help="embed(삽입) / extract(추출)"
    )
    parser.add_argument("--image", required=True, help="입력(또는 스테고) 이미지 경로")
    parser.add_argument("--message", help="삽입할 메시지( embed 모드 )")
    parser.add_argument("--output", help="출력 스테고 이미지 경로( embed 모드 )")
    parser.add_argument(
        "--levels", type=int, default=3, help="DTCWT 분해 레벨 (기본: 3)"
    )
    parser.add_argument(
        "--strength", type=float, default=30.0, help="최소 실수부 크기(기본: 30.0)"
    )
    parser.add_argument(
        "--redundancy", type=int, default=8, help="비트당 중복 삽입 횟수 (기본: 8)"
    )
    parser.add_argument(
        "--key", type=int, default=12345, help="좌표 난수 시드 (송수신 동일)"
    )
    parser.add_argument(
        "--stride", type=int, default=2, help="좌표 샘플링 간격 (기본: 2)"
    )
    parser.add_argument("--debug", action="store_true", help="추출 디버그 정보 출력")
    parser.add_argument("--verify", action="store_true", help="삽입 후 검증 수행")

    args = parser.parse_args()

    if args.mode == "embed":
        if not args.message or not args.output:
            print("오류: embed 모드에서는 --message 와 --output 이 필요합니다.")
            return
        embed_message(
            image_path=args.image,
            message=args.message,
            output_path=args.output,
            levels=args.levels,
            strength=args.strength,
            redundancy=args.redundancy,
            key=args.key,
            stride=args.stride,
            verify=args.verify,
        )
    else:
        msg = extract_message(
            image_path=args.image,
            levels=args.levels,
            redundancy=args.redundancy,
            key=args.key,
            stride=args.stride,
            debug=args.debug,
        )
        print(f"추출된 메시지: {msg}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 68)
        print("DTCWT 스테가노그래피 (대칭 방향 결합 + 관계 부호 + 다수결)")
        print("=" * 68)
        print("\n특징")
        print("- 대칭 방향쌍(±θ)에서 관계 부호 r=sgn(Re(c+θ))*sgn(Re(c−θ))로 비트 표현")
        print("- 비밀키 기반 좌표 시퀀스 + 중복 삽입(redundancy) + 가중 다수결 복호")
        print("- strength로 최소 실수부 크기 보장(역변환 노이즈 내성 개선, 기본 30.0)")
        print("\n예시")
        print("1) 삽입:")
        print(
            "   python tmp.py embed --image input.png --message 'Hello' --output stego.png"
        )
        print("      --levels 3 --strength 30 --redundancy 8 --key 12345 --stride 2")
        print("2) 추출:")
        print(
            "   python script.py extract --image stego.png --levels 3 --redundancy 8 --key 12345 --stride 2"
        )
        print("\n설치:")
        print("   pip install numpy pillow dtcwt")
        print("\n주의:")
        print(
            "- 삽입/추출 시 --levels, --redundancy, --key, --stride 는 동일해야 합니다."
        )
        print("- PNG 등 무손실 포맷을 권장합니다.")
        print("=" * 68)
    else:
        main()
