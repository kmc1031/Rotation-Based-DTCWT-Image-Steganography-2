import numpy as np
from dtcwt import Transform2d
from dtcwt.numpy import Pyramid
from PIL import Image
import argparse


def text_to_bits(text):
    """텍스트를 비트 문자열로 변환"""
    bits = "".join(format(ord(c), "08b") for c in text)
    return bits


def bits_to_text(bits):
    """비트 문자열을 텍스트로 변환"""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i : i + 8]
        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))
    return "".join(chars)


def embed_message(image_path, message, output_path, levels=3, strength=10):
    """
    DTCWT를 사용하여 이미지에 메시지 삽입

    Parameters:
    - image_path: 원본 이미지 경로
    - message: 삽입할 텍스트 메시지
    - output_path: 결과 이미지 저장 경로
    - levels: DTCWT 분해 레벨 (기본값: 3)
    - strength: 메시지 삽입 강도 (기본값: 10)
    """
    # 이미지 로드 (Grayscale로 변환)
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float64)

    # 메시지를 비트로 변환하고 종료 마커 추가
    message_with_delimiter = message + "###END###"
    bits = text_to_bits(message_with_delimiter)
    message_length = len(bits)

    # 메시지 길이를 32비트로 인코딩
    length_bits = format(message_length, "032b")
    total_bits = length_bits + bits

    # Grayscale 이미지에 대해 DTCWT 수행
    gray_channel = img_array

    # DTCWT 변환 객체 생성
    transform = Transform2d()

    # DTCWT 변환 수행
    coeffs = transform.forward(gray_channel, nlevels=levels)

    # 저주파 계수와 고주파 계수
    lowpass = coeffs.lowpass.copy()
    highpass = [band.copy() for band in coeffs.highpasses]

    # 첫 번째 레벨의 고주파 계수 (가장 세밀한 디테일)
    # highpass[0]의 형태: (height, width, 6) - 6개 방향
    first_level = highpass[0]

    # 첫 번째 방향의 실수부를 평탄화
    coeff_real = np.real(first_level[:, :, 0]).flatten()

    # 메시지를 삽입할 공간이 충분한지 확인
    if len(total_bits) > len(coeff_real):
        raise ValueError(
            f"메시지가 너무 깁니다. 최대 {(len(coeff_real) - 32) // 8} 문자까지 가능합니다."
        )

    # 메시지 비트를 고주파 계수의 부호로 삽입 (강한 강도)
    for i, bit in enumerate(total_bits):
        magnitude = max(abs(coeff_real[i]), strength)

        if bit == "1":
            coeff_real[i] = magnitude  # 확실히 양수
        else:
            coeff_real[i] = -magnitude  # 확실히 음수

    # 수정된 계수를 원래 형태로 복원
    first_level[:, :, 0] = coeff_real.reshape(
        first_level[:, :, 0].shape
    ) + 1j * np.imag(first_level[:, :, 0])
    highpass[0] = first_level

    # 역 DTCWT 수행
    gray_channel_modified = transform.inverse(Pyramid(lowpass, highpass))

    # 원본 크기 유지
    h, w = gray_channel.shape
    gray_channel_modified = gray_channel_modified[:h, :w]

    # 수정된 Grayscale 채널을 이미지에 적용
    img_array = gray_channel_modified

    # 픽셀 값을 0-255 범위로 클리핑
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # 결과 이미지 저장 (PNG로 무손실 저장)
    result_img = Image.fromarray(img_array, mode="L")
    result_img.save(output_path, "PNG")

    print(f"메시지가 성공적으로 삽입되었습니다: {output_path}")
    print(f"삽입된 메시지 길이: {len(message)} 문자")
    print(f"사용된 계수: {len(total_bits)} / {len(coeff_real)}")
    print(f"DTCWT 레벨: {levels}")
    print(f"삽입 강도: {strength}")


def extract_message(image_path, levels=3, debug=False):
    """
    DTCWT를 사용하여 이미지에서 메시지 추출

    Parameters:
    - image_path: 메시지가 포함된 이미지 경로
    - levels: DTCWT 분해 레벨 (기본값: 3)
    - debug: 디버그 정보 출력 여부

    Returns:
    - 추출된 텍스트 메시지
    """
    # 이미지 로드 (Grayscale로 변환)
    img = Image.open(image_path).convert("L")
    img_array = np.array(img, dtype=np.float64)

    # Grayscale 이미지에서 DTCWT 수행
    gray_channel = img_array

    # DTCWT 변환 객체 생성
    transform = Transform2d()

    # DTCWT 변환 수행
    coeffs = transform.forward(gray_channel, nlevels=levels)

    # 첫 번째 레벨의 고주파 계수
    first_level = coeffs.highpasses[0]
    coeff_real = np.real(first_level[:, :, 0]).flatten()

    # 먼저 메시지 길이 추출 (첫 32비트)
    length_bits = ""
    for i in range(32):
        if coeff_real[i] >= 0:
            length_bits += "1"
        else:
            length_bits += "0"

        if debug and i < 5:
            print(f"디버그 계수[{i}]: value={coeff_real[i]:.4f}, bit={length_bits[-1]}")

    message_length = int(length_bits, 2)

    if debug:
        print(f"디버그: 추출된 길이 비트: {length_bits}")
        print(f"디버그: 메시지 길이: {message_length}")
        print(f"디버그: 사용 가능한 계수: {len(coeff_real)}")

    # 메시지 길이가 비정상적으로 크면 오류
    if message_length > len(coeff_real) * 8 or message_length < 0:
        return f"오류: 메시지를 찾을 수 없습니다. (길이: {message_length})"

    # 실제 메시지 비트 추출
    bits = ""
    for i in range(32, 32 + message_length):
        if i >= len(coeff_real):
            break

        if coeff_real[i] >= 0:
            bits += "1"
        else:
            bits += "0"

    if debug:
        print(f"디버그: 추출된 메시지 비트 (처음 64비트): {bits[:64]}")

    # 비트를 텍스트로 변환
    message = bits_to_text(bits)

    # 종료 마커 찾기
    delimiter_pos = message.find("###END###")
    if delimiter_pos != -1:
        message = message[:delimiter_pos]

    return message


def main():
    parser = argparse.ArgumentParser(description="DTCWT 기반 스테가노그래피")
    parser.add_argument(
        "mode",
        choices=["embed", "extract"],
        help="작동 모드: embed (삽입) 또는 extract (추출)",
    )
    parser.add_argument("--image", required=True, help="이미지 파일 경로")
    parser.add_argument("--message", help="삽입할 메시지 (embed 모드에서 필요)")
    parser.add_argument("--output", help="출력 이미지 경로 (embed 모드에서 필요)")
    parser.add_argument(
        "--levels", type=int, default=3, help="DTCWT 분해 레벨 (기본값: 3)"
    )
    parser.add_argument(
        "--strength", type=float, default=10, help="삽입 강도 (기본값: 10, 범위: 5-50)"
    )
    parser.add_argument("--debug", action="store_true", help="디버그 정보 출력")

    args = parser.parse_args()

    if args.mode == "embed":
        if not args.message or not args.output:
            print("오류: embed 모드에서는 --message와 --output이 필요합니다.")
            return
        embed_message(args.image, args.message, args.output, args.levels, args.strength)

    elif args.mode == "extract":
        message = extract_message(args.image, args.levels, args.debug)
        print(f"추출된 메시지: {message}")


if __name__ == "__main__":
    # 명령줄 인수가 없으면 예제 실행
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("DTCWT 기반 스테가노그래피 프로그램")
        print("=" * 60)
        print("\nDTCWT (Dual-Tree Complex Wavelet Transform)는:")
        print("- 더 나은 방향 선택성 제공")
        print("- 시프트 불변성 향상")
        print("- 기존 DWT보다 강건한 메시지 은닉")
        print("\n개선된 알고리즘:")
        print("- 고주파 계수의 부호(+/-)로 비트 인코딩")
        print("- 양자화 방식보다 더 강건함")
        print("\n사용 예시:")
        print("\n1. 메시지 삽입:")
        print(
            "   python script.py embed --image input.png --message 'Hello World!' --output output.png"
        )
        print("\n2. 메시지 추출:")
        print("   python script.py extract --image output.png")
        print("\n3. 레벨 조절 (선택사항):")
        print(
            "   python script.py embed --image input.png --message 'Secret' --output output.png --levels 4"
        )
        print("   python script.py extract --image output.png --levels 4")
        print("\n필요한 라이브러리:")
        print("   pip install numpy pillow dtcwt")
        print("\n참고:")
        print("   - 반드시 PNG 형식으로 저장하세요 (무손실)")
        print("   - 삽입과 추출 시 동일한 levels 값 사용 필요")
        print("=" * 60)
    else:
        main()
