import matplotlib.pyplot as plt
import numpy as np
from simulation import run_headless_simulation

def evaluate_planner():
    scenarios = ['Empty', 'Single', 'Corridor', 'U-Trap', 'Complex']
    results = []

    print(f"{'Scenario':<15} | {'Mode':<10} | {'Success':<8} | {'Steps':<6} | {'Length':<8} | {'MinClear':<8}")
    print("-" * 75)

    for name in scenarios:
        res_apf = run_headless_simulation(name, use_hybrid=False, max_steps=600)
        results.append(res_apf)
        print(f"{res_apf['scenario']:<15} | {res_apf['mode']:<10} | {str(res_apf['success']):<8} | {res_apf['steps']:<6} | {res_apf['length']:<8.2f} | {res_apf['min_clearance']:<8.2f}")

        res_hybrid = run_headless_simulation(name, use_hybrid=True, max_steps=600)
        results.append(res_hybrid)
        print(f"{res_hybrid['scenario']:<15} | {res_hybrid['mode']:<10} | {str(res_hybrid['success']):<8} | {res_hybrid['steps']:<6} | {res_hybrid['length']:<8.2f} | {res_hybrid['min_clearance']:<8.2f}")

    return results

def plot_evaluation_results(results):
    scenarios = sorted(list(set(r['scenario'] for r in results)))
    scenario_order = ['Empty', 'Single', 'Corridor', 'U-Trap', 'Complex']
    scenarios = [s for s in scenario_order if s in scenarios]
    
    apf_lengths = []
    hybrid_lengths = []
    apf_success = []
    hybrid_success = []

    for s in scenarios:
        apf = next(r for r in results if r['scenario'] == s and r['mode'] == 'APF')
        apf_lengths.append(apf['length'])
        apf_success.append(apf['success'])

        hybrid = next(r for r in results if r['scenario'] == s and r['mode'] == 'Hybrid')
        hybrid_lengths.append(hybrid['length'])
        hybrid_success.append(hybrid['success'])

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, apf_lengths, width, label='APF', color='skyblue')
    rects2 = ax.bar(x + width/2, hybrid_lengths, width, label='Hybrid', color='orange')

    for i, rect in enumerate(rects1):
        if not apf_success[i]:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 1, 'FAIL',
                    ha='center', va='bottom', color='red', fontweight='bold', rotation=90)
    
    for i, rect in enumerate(rects2):
        if not hybrid_success[i]:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 1, 'FAIL',
                    ha='center', va='bottom', color='red', fontweight='bold', rotation=90)

    ax.set_ylabel('Path Length')
    ax.set_title('Planner Performance: APF vs Hybrid')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close(fig)

if __name__ == "__main__":
    data = evaluate_planner()
    plot_evaluation_results(data)
