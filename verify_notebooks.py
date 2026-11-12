"""Validate the saved deliverables, without repeating model training."""
import re
from pathlib import Path
import nbformat

root = Path(__file__).parent / 'assignment2'
expected = ['BatchNormalization', 'Dropout', 'ConvolutionalNetworks',
            'PyTorch', 'RNN_Captioning_pytorch']
total_cells = total_figures = 0
for name in expected:
    nb = nbformat.read(root / (name + '.ipynb'), as_version=4)
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code), name + ': unexecuted cells'
    assert not any(output.output_type == 'error' for cell in code for output in cell.outputs), name + ': error output'
    assert not any('впишите ответ здесь' in cell.source.lower() for cell in nb.cells), name + ': unanswered question'
    figures = sum('image/png' in output.get('data', {}) for cell in code for output in cell.outputs)
    assert figures > 0, name + ': no saved figures'
    total_cells += len(code)
    total_figures += figures
    print(f'{name}: {len(code)} code cells executed, {figures} figures, no errors')
    if name == 'PyTorch':
        text = ''.join(output.get('text', '') for cell in code for output in cell.outputs)
        match = re.search(r'Best validation accuracy: ([0-9.]+)%', text)
        assert match and float(match[1]) >= 70, 'Validation target not reached'
        assert 'Epoch 10/10:' in text, 'Ten-epoch experiment incomplete'
        assert re.search(r'из 10000 \([0-9.]+%\)', text), 'Final test result missing'
        print(match[0])
print(f'PASS: {total_cells} executed code cells and {total_figures} saved figures')
