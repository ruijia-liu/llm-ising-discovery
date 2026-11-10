"""Publication figures for completed curriculum audits, never student effect plots."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve the existing figures and their provenance.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ['MPLCONFIGDIR'] = str(Path('results/.matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, NullLocator, ScalarFormatter
    from PIL import Image

    paths = {'teacher': 'results/teacher_research_content/audit.json',
             'exposure': 'results/publication_exposure_audit/audit.json'}
    teacher, exposure = (read(paths[k]) for k in ('teacher', 'exposure'))
    for audit in (teacher, exposure):
        for path, digest in audit['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('A curriculum source differs from the plotted audit: '+path)
    if (not teacher['not_student_results'] or teacher['test_worlds_opened']
            or exposure['reserved_test_answers_opened']):
        raise ValueError('These figures accept audited teaching/development evidence only.')
    rows = {(r['split'], r['sigma']): r for r in teacher['summary']}
    if len(rows) != 4 or set(rows) != {(split, sigma) for split in ('train', 'validation') for sigma in (0., 1.)}:
        raise ValueError('Missing or duplicated curriculum noise/split cells.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'ising-curriculum-audits-v1', 'savefig.facecolor': 'white'})
    colors = ('#2673A3', '#D58A2F')
    files, plotted = [], {'teacher': [], 'exposure': exposure['cumulative_exposure']}

    def save(fig, name):
        for suffix in ('png', 'svg', 'pdf'):
            path = output/(name+'.'+suffix)
            options = {'dpi': 250} if suffix == 'png' else {}
            fig.savefig(path, bbox_inches='tight', pad_inches=.12, **options)
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.76, wspace=.29)
    fig.suptitle('Teacher process content', fontsize=17, x=.5, y=.985)
    fig.text(.5, .923, 'Complete curriculum audit; these are teacher trajectories, not student outcomes',
             ha='center', fontsize=10, color='#4B5563')
    panels = (('first8_correct_support', '(a) Correct structure after 8 measurements'),
              ('worlds_with_structure_update', '(b) At least one structural revision'))
    for ax, (metric, title) in zip(axes, panels):
        for index, sigma in enumerate((0., 1.)):
            selected = [rows[split, sigma] for split in ('train', 'validation')]
            if any(not 0 <= r[metric] <= r['worlds'] or r['worlds'] <= 0 for r in selected):
                raise ValueError('Invalid count/denominator in a figure.')
            values = [r[metric]/r['worlds'] for r in selected]
            positions = [group+(-.17 if index == 0 else .17) for group in (0, 1)]
            bars = ax.bar(positions, values, width=.30, color=colors[index], edgecolor='#344454', linewidth=.5,
                          hatch=None if index == 0 else '//', zorder=3,
                          label=('Noiseless ($\\sigma=0$)' if index == 0 else 'Noisy ($\\sigma=1$)'))
            labels = [f"{r[metric]}/{r['worlds']}\n{100*v:.1f}%" for r, v in zip(selected, values)]
            ax.bar_label(bars, labels=labels, padding=5, fontsize=9)
            plotted['teacher'] += [{'metric': metric, 'split': r['split'], 'sigma': sigma,
                'count': r[metric], 'worlds': r['worlds'], 'proportion': value} for r, value in zip(selected, values)]
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_xticks([0, 1], ['Training\n300 worlds', 'Development\n60 worlds'])
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, .25, .5, .75, 1.])
        ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
        ax.set_ylabel('Proportion of worlds')
        ax.grid(axis='y', color='#DFE4E8', linewidth=.7, zorder=0)
        ax.tick_params(axis='x', length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .885), ncol=2, frameon=False)
    fig.text(.075, .063, 'Each world has a separate 32-measurement run at each noise level. All teacher failures are retained.', fontsize=9)
    fig.text(.075, .025, 'A structural revision need not improve the hypothesis. Bars show complete-corpus counts, not inferred teaching effects.', fontsize=9)
    save(fig, 'teacher_process_content')

    stages = exposure['cumulative_exposure']
    names = ['Final training answers only', '+ All formal C training content', '+ All formal C development content',
             '+ Earlier training curriculum', '+ Earlier development curriculum', '+ Coverage-diagnostic training',
             '+ Coverage-diagnostic development']
    expected_order = ('formal_training_final_answers_only', 'formal_training_all_C_content',
        'plus_formal_validation_all_C_content', 'plus_legacy_training', 'plus_legacy_validation',
        'plus_coverage_training', 'plus_coverage_validation')
    if tuple(r['scope'] for r in stages) != expected_order:
        raise ValueError('Exposure rows do not match their figure labels.')
    values = [r['eligible_unseen_pairs'] for r in stages]
    if any(a < b or b <= 0 for a, b in zip(values, values[1:])):
        raise ValueError('Cumulative exclusions must not add eligible mechanisms.')
    fig, ax = plt.subplots(figsize=(10.8, 5.25))
    fig.subplots_adjust(left=.365, right=.935, top=.80, bottom=.23)
    fig.suptitle('Mechanism exposure extends beyond final answers', fontsize=16, y=.985)
    fig.text(.5, .918, 'Eligible two-term combinations after cumulative exclusions', ha='center', fontsize=11, color='#4B5563')
    for y, value in enumerate(values):
        ax.hlines(y, 8, value, color='#D7E4ED', linewidth=2, zorder=1)
        ax.scatter(value, y, s=60, color=colors[0], edgecolor='white', linewidth=.7, zorder=3)
        ax.annotate(f'{value:,}', (value, y), xytext=(9, 0), textcoords='offset points',
                    va='center', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(8, 2600)
    ax.set_xticks([10, 50, 100, 500, 2000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names)-.55, -.45)
    ax.set_xlabel('Eligible distinct mechanisms (logarithmic scale)', labelpad=9)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=11)
    ax.grid(axis='x', color='#E6EAED', linewidth=.7, zorder=0)
    fig.text(.045, .081, f"{exposure['training_components']} components in training answers define {exposure['possible_distinct_pairs']:,} pairs. Exclusions include incorrect candidates.", fontsize=9.5)
    fig.text(.045, .038, f"The {exposure['reserved_combination_worlds']} reserved combination worlds represent {exposure['reserved_distinct_combinations']} distinct structures. Exclusion order affects stage attribution.", fontsize=9.5)
    save(fig, 'mechanism_exposure')

    (output/'plotted_values.json').write_text(json.dumps(plotted, indent=2)+'\n', encoding='utf-8')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    manifest = {'completed': True, 'scope': 'completed_curriculum_audits_only_no_student_effects',
        'formal_test_results_used': False, 'source_sha256': {p: sha(p) for p in paths.values()},
        'plot_script_sha256': sha(__file__), 'packages': packages,
        'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files},
        'png_sizes': {p.name: list(Image.open(p).size) for p in files if p.suffix == '.png'},
        'plotted_values_sha256': sha(output/'plotted_values.json')}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (output/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k, v in sorted(packages.items()) if k.lower() != 'pip'), encoding='utf-8')
    print(json.dumps({'output': str(output), 'figures': 2, 'export_files': len(files), 'scope': manifest['scope']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='paper/figures/curriculum_audits_v2')
    run(parser.parse_args().output)
