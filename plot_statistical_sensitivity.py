"""Plot synthetic statistical sensitivity; never a student-effect figure."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from plot_curriculum_audits import read, sha


def run(source, output):
    source, output = Path(source), Path(output)
    audit = read(source/'result.json')
    if (not audit['completed'] or audit['formal_tests_used'] or audit['student_results_used']
            or audit['scope'] != 'synthetic_design_sensitivity_not_student_results'):
        raise ValueError('Only the complete, explicitly synthetic audit can enter this figure.')
    for path, digest in audit['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Synthetic audit source changed: '+path)
    rows = audit['scenarios']
    if len(rows) != 12 or {r['scenario'] for r in rows} != set(range(12)):
        raise ValueError('The full twelve-scenario grid is required.')
    for row in rows:
        if sha(source/row['saved_arrays']) != row['arrays_sha256']:
            raise ValueError('Synthetic outcome arrays changed.')
        for name in ('advantage_rule', 'coverage'):
            count = row[name]
            if count['denominator'] != 512 or count['fraction'] != count['count']/512:
                raise ValueError('Plotted rate does not match its saved count.')
    if output.exists():
        raise ValueError('Preserve existing figures; choose a new output directory.')
    output.mkdir(parents=True)
    (output/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ['MPLCONFIGDIR'] = str(Path('results/.matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from PIL import Image
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'ising-statistical-sensitivity-v1', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3))
    fig.subplots_adjust(left=.075, right=.98, bottom=.24, top=.76, wspace=.28)
    fig.suptitle('Statistical sensitivity under synthetic outcomes', fontsize=17, y=.98)
    fig.text(.5, .925, 'Illustrative dependence assumptions; no student results or heldout-world data',
             ha='center', color='#4B5563')
    colors = ('#2673A3', '#C47719')
    for dependence, color, marker, label in zip(('independent_cells', 'structure_shared'), colors,
                                               ('o', 's'), ('Independent cells', 'Identical outcomes within each structure')):
        selected = [r for r in rows if r['dependence'] == dependence]
        for panel, metric in ((0, 'advantage_rule'), (1, 'coverage')):
            shown = [r for r in selected if r['p_B'] == .2] if panel == 0 else selected
            x = [100*r['true_difference'] for r in shown] if panel == 0 else [i+(-.09 if panel == 1 and dependence == 'independent_cells' else .09) for i in range(6)]
            axes[panel].errorbar(x, [r[metric]['fraction'] for r in shown],
                                yerr=[r[metric]['monte_carlo_standard_error'] for r in shown],
                                marker=marker, color=color, linestyle='-' if panel == 0 else 'none',
                                linewidth=1.7, capsize=3, markersize=5, label=label)
    axes[0].set(title='(a) Meets the frozen advantage rule', xlabel='Assumed true C minus B (percentage points)',
                ylabel='Proportion of synthetic studies', ylim=(-.035, 1.04), xticks=[0, 5, 10, 20])
    axes[0].text(.025, .965, 'B success probability = 0.20', transform=axes[0].transAxes, va='top', fontsize=9)
    axes[1].set(title='(b) Coverage of the known difference', ylabel='Proportion of synthetic intervals',
                ylim=(.80, 1.035), xticks=range(6), xticklabels=['.20 / 0', '.20 / 5', '.20 / 10', '.20 / 20', '.02 / 0', '.02 / 5'],
                xlabel='B success probability / true difference (pp)')
    axes[1].tick_params(axis='x', labelsize=8.5)
    axes[1].axhline(.95, color='#636B73', linewidth=1, linestyle='--', zorder=0)
    axes[1].text(5.4, .946, 'nominal 95%', ha='right', va='top', fontsize=9, color='#636B73')
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
        ax.grid(axis='y', linewidth=.6, color='#E6EAED')
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.5, .88), frameon=False)
    fig.text(.075, .087, '512 studies per scenario; bars show one Monte Carlo SE, not uncertainty in an observed teaching effect.', fontsize=9)
    fig.text(.075, .044, '11 combination + 50 fourth-order structures; 3 fixed seeds. Synthetic assumptions do not guarantee real-study power or coverage.', fontsize=9)
    files = []
    for suffix in ('png', 'svg', 'pdf'):
        path = output/('statistical_sensitivity.'+suffix)
        fig.savefig(path, bbox_inches='tight', pad_inches=.12, **({'dpi': 250} if suffix == 'png' else {}))
        files.append(path)
    plt.close(fig)
    (output/'plotted_values.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    manifest = {'completed': True, 'scope': 'synthetic_statistical_sensitivity_no_student_outcomes',
                'source_result': str(source/'result.json'), 'source_result_sha256': sha(source/'result.json'),
                'plot_script_sha256': sha(__file__), 'packages': packages,
                'plotted_values_sha256': sha(output/'plotted_values.json'),
                'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files},
                'png_sizes': {p.name: list(Image.open(p).size) for p in files if p.suffix == '.png'}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (output/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k, v in sorted(packages.items()) if k.lower() != 'pip'), encoding='utf-8')
    print(output/'statistical_sensitivity.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/statistical_sensitivity_v1')
    parser.add_argument('--output', default='paper/figures/statistical_sensitivity_v1')
    args = parser.parse_args()
    run(args.source, args.output)
