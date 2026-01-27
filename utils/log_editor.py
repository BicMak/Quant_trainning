"""
Quantization Layer Profiler Log Editor

모듈화된 프로파일러 결과 저장 유틸리티:
- 레이어별 profiler 결과 저장
- prof.get_statistic()의 qsnr 활용
- CSV 테이블 형태로 통계 저장
- 히스토그램 시각화 (레이어 이름 제목)
- 날짜시간 폴더 생성
"""

import torch
from pathlib import Path
from datetime import datetime
import json
import csv
from typing import List, Tuple, Optional, Dict, Any

from models.ptq.layer_profiler.profiler import profiler as ProfilerClass


def save_layer_profiler_results(
    profiler_list: List[Tuple[str, ProfilerClass]],
    base_log_dir: str = "log",
    prefix: str = "layer"
) -> Dict[str, Any]:
    """
    레이어별 profiler 결과를 저장합니다.

    Args:
        profiler_list: [(layer_name, profiler_object), ...] 형태의 리스트
        base_log_dir: 기본 로그 디렉토리 (기본값: "log")
        prefix: 파일 prefix (기본값: "layer")

    Returns:
        dict: 저장된 파일 경로들
    """
    # 날짜시간 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(base_log_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {
        'directory': str(save_dir),
        'csv': None,
        'histograms': [],
        'summary': None
    }

    print(f"\n[Saving Profiler Results]")
    print(f"  Save directory: {save_dir}")
    print(f"  Total layers: {len(profiler_list)}")

    # 통계 데이터 수집 (CSV용)
    stats_rows = []
    summary_data = {
        'timestamp': timestamp,
        'total_layers': len(profiler_list),
        'layers': {}
    }

    for layer_name, prof in profiler_list:
        if prof is None:
            continue

        print(f"\n  Processing: {layer_name}")

        # 통계 데이터 가져오기 (qsnr 포함)
        stat_data = prof.get_statistic()
        hist_data = prof.get_hist()

        layer_info = {'name': layer_name}

        # 통계 데이터 처리
        if stat_data is not None:
            row = {'layer_name': layer_name}

            for k, v in stat_data.items():
                if isinstance(v, torch.Tensor):
                    row[k] = v.item() if v.numel() == 1 else float(v.mean())
                else:
                    row[k] = v

            stats_rows.append(row)
            layer_info['statistics'] = row

            # qsnr 출력 (prof.get_statistic()에서 가져온 값)
            if 'qsnr' in stat_data:
                qsnr_val = stat_data['qsnr']
                if isinstance(qsnr_val, torch.Tensor):
                    qsnr_val = qsnr_val.item()
                print(f"    - QSNR: {qsnr_val:.2f} dB")

        # 히스토그램 처리
        if hist_data is not None:
            layer_info['histogram'] = {
                'kl_divergence': hist_data.get('kl_divergence'),
                'js_divergence': hist_data.get('js_divergence'),
                'original_range': hist_data.get('original_range'),
                'quantized_range': hist_data.get('quantized_range'),
            }

            # 히스토그램 시각화 저장
            hist_file = _save_histogram(
                hist_data=hist_data,
                stat_data=stat_data,
                layer_name=layer_name,
                save_dir=save_dir,
                prefix=prefix
            )
            if hist_file:
                saved_files['histograms'].append(str(hist_file))
                print(f"    - Histogram saved: {hist_file.name}")

        summary_data['layers'][layer_name] = layer_info

    # CSV 저장
    if stats_rows:
        csv_file = save_dir / f"{prefix}_statistics.csv"
        _save_stats_csv(stats_rows, csv_file)
        saved_files['csv'] = str(csv_file)
        print(f"\n  Statistics CSV saved: {csv_file.name}")

    # Summary JSON 저장
    summary_file = save_dir / f"{prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    saved_files['summary'] = str(summary_file)
    print(f"  Summary JSON saved: {summary_file.name}")

    # 콘솔에 요약 테이블 출력
    _print_summary_table(stats_rows)

    return saved_files


def _save_stats_csv(stats_rows: List[Dict], csv_file: Path):
    """통계 데이터를 CSV로 저장"""
    if not stats_rows:
        return

    # 모든 컬럼 수집
    all_keys = set()
    for row in stats_rows:
        all_keys.update(row.keys())

    # 컬럼 순서 정의 (중요한 것 먼저)
    priority_cols = ['layer_name', 'qsnr', 'mse', 'original_min', 'original_max',
                     'original_mean', 'original_std', 'quantized_min', 'quantized_max',
                     'quantized_mean', 'quantized_std']

    fieldnames = []
    for col in priority_cols:
        if col in all_keys:
            fieldnames.append(col)
            all_keys.discard(col)
    fieldnames.extend(sorted(all_keys))

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats_rows:
            # 숫자 포맷팅
            formatted_row = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted_row[k] = f"{v:.6f}"
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)


def _save_histogram(
    hist_data: Dict,
    stat_data: Optional[Dict],
    layer_name: str,
    save_dir: Path,
    prefix: str
) -> Optional[Path]:
    """히스토그램 시각화 저장"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        # 데이터 추출
        orig_hist = hist_data['original_hist'].numpy()
        quant_hist = hist_data['quantized_hist'].numpy()
        orig_range = hist_data['original_range']
        quant_range = hist_data['quantized_range']

        # 공통 범위로 bins 생성
        min_val = min(orig_range[0], quant_range[0])
        max_val = max(orig_range[1], quant_range[1])
        bins = np.linspace(min_val, max_val, len(orig_hist) + 1)
        bin_width = (max_val - min_val) / len(orig_hist)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Figure 생성 - 3개 subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Original Distribution
        axes[0].bar(bin_centers, orig_hist, width=bin_width * 0.9,
                   alpha=0.8, color='blue', edgecolor='darkblue')
        axes[0].set_title(f'Original Distribution', fontsize=12)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')
        axes[0].axvline(x=orig_range[0], color='red', linestyle='--', alpha=0.7, label=f'min={orig_range[0]:.4f}')
        axes[0].axvline(x=orig_range[1], color='red', linestyle='--', alpha=0.7, label=f'max={orig_range[1]:.4f}')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 2. Quantized Distribution
        axes[1].bar(bin_centers, quant_hist, width=bin_width * 0.9,
                   alpha=0.8, color='orange', edgecolor='darkorange')
        axes[1].set_title(f'Quantized Distribution', fontsize=12)
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Count')
        axes[1].axvline(x=quant_range[0], color='red', linestyle='--', alpha=0.7, label=f'min={quant_range[0]:.4f}')
        axes[1].axvline(x=quant_range[1], color='red', linestyle='--', alpha=0.7, label=f'max={quant_range[1]:.4f}')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # 3. Overlay Comparison
        kl_div = hist_data.get("kl_divergence", 0)
        js_div = hist_data.get("js_divergence", 0)
        axes[2].bar(bin_centers, orig_hist, width=bin_width * 0.9,
                   alpha=0.5, color='blue', edgecolor='darkblue', label='Original')
        axes[2].bar(bin_centers, quant_hist, width=bin_width * 0.9,
                   alpha=0.5, color='orange', edgecolor='darkorange', label='Quantized')
        axes[2].set_title(f'Overlay Comparison\nKL={kl_div:.4f}, JS={js_div:.4f}', fontsize=12)
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Count')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # 전체 제목: 레이어 이름 + QSNR
        title = f'{layer_name}'
        if stat_data and 'qsnr' in stat_data:
            qsnr_val = stat_data['qsnr']
            if isinstance(qsnr_val, torch.Tensor):
                qsnr_val = qsnr_val.item()
            title += f' (QSNR: {qsnr_val:.2f} dB)'

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 파일명에서 특수문자 제거
        safe_layer_name = layer_name.replace('/', '_').replace('\\', '_')
        hist_file = save_dir / f"{prefix}_{safe_layer_name}_hist.png"
        plt.savefig(hist_file, dpi=150, bbox_inches='tight')
        plt.close()

        return hist_file

    except ImportError:
        print(f"    - Histogram skipped (matplotlib not available)")
        return None
    except Exception as e:
        print(f"    - Histogram error: {e}")
        return None


def _print_summary_table(stats_rows: List[Dict]):
    """콘솔에 요약 테이블 출력"""
    if not stats_rows:
        return

    print(f"\n" + "=" * 70)
    print("Layer-wise Statistics Summary")
    print("=" * 70)
    print(f"{'Layer':<35} {'QSNR (dB)':>12} {'MSE':>15}")
    print("-" * 70)

    for row in stats_rows:
        layer_name = row.get('layer_name', 'N/A')
        qsnr = row.get('qsnr', 'N/A')
        mse = row.get('mse', 'N/A')

        # 긴 레이어 이름 자르기
        if len(layer_name) > 33:
            layer_name = layer_name[:30] + "..."

        qsnr_str = f"{qsnr:.2f}" if isinstance(qsnr, (int, float)) else str(qsnr)
        mse_str = f"{mse:.6e}" if isinstance(mse, (int, float)) else str(mse)

        print(f"{layer_name:<35} {qsnr_str:>12} {mse_str:>15}")

    print("=" * 70)


# ============================================================
# Convenience functions
# ============================================================

def save_qvit_profiler_results(qvit, base_log_dir: str = "log", prefix: str = "qvit"):
    """
    QVit 모델의 모든 레이어 profiler 결과를 저장합니다.

    Args:
        qvit: QVit 인스턴스
        base_log_dir: 기본 로그 디렉토리
        prefix: 파일 prefix

    Returns:
        dict: 저장된 파일 경로들
    """
    profiler_list = qvit.get_profiler_list()
    return save_layer_profiler_results(profiler_list, base_log_dir, prefix)


def save_qblock_profiler_results(qblock, base_log_dir: str = "log", prefix: str = "qblock"):
    """
    QVitBlock의 모든 레이어 profiler 결과를 저장합니다.

    Args:
        qblock: QVitBlock 인스턴스
        base_log_dir: 기본 로그 디렉토리
        prefix: 파일 prefix

    Returns:
        dict: 저장된 파일 경로들
    """
    profiler_list = qblock.get_profiler_list()
    return save_layer_profiler_results(profiler_list, base_log_dir, prefix)


if __name__ == '__main__':
    # 테스트 코드
    print("log_editor module - use save_layer_profiler_results() or convenience functions")
    print("  - save_qvit_profiler_results(qvit)")
    print("  - save_qblock_profiler_results(qblock)")
    print("  - save_layer_profiler_results(profiler_list)")