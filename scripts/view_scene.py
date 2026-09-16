"""View a random scene or save a preview frame."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description="随机场景预览（仅运动，无战斗结算）")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=0, help='预先推进的步数')
    parser.add_argument('--save', type=Path, help='保存图片并退出')
    parser.add_argument('--interval', type=int, default=120, help='动画间隔，毫秒')
    args = parser.parse_args()
    if args.steps < 0 or args.interval <= 0:
        parser.error('steps 必须非负，interval 必须为正')
    if args.save:
        import matplotlib
        matplotlib.use('Agg')
    from fofe_mmapppo.visualization.live_viewer import LiveViewer
    viewer = LiveViewer(seed=args.seed, interval_ms=args.interval)
    for _ in range(args.steps):
        viewer._advance_scene_only()
    if args.save:
        viewer.paused = True
        viewer.renderer.draw()
        viewer.fig.canvas.draw()
        args.save.parent.mkdir(parents=True, exist_ok=True)
        viewer.fig.savefig(args.save, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(viewer.fig)
        print(f'Saved {args.save}')
    else:
        viewer.show()


if __name__ == '__main__':
    main()
