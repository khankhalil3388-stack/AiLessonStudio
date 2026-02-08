// Interactive JavaScript for AI Lesson Studio

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();

    // Initialize copy buttons
    initCopyButtons();

    // Initialize diagram interactivity
    initDiagrams();

    // Initialize progress animations
    initProgressAnimations();
});

function initTooltips() {
    // Add tooltips to elements with data-tooltip attribute
    const tooltipElements = document.querySelectorAll('[data-tooltip]');

    tooltipElements.forEach(el => {
        const tooltipText = el.getAttribute('data-tooltip');

        el.addEventListener('mouseenter', function(e) {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = tooltipText;
            tooltip.style.position = 'absolute';
            tooltip.style.background = '#1F2937';
            tooltip.style.color = 'white';
            tooltip.style.padding = '8px 12px';
            tooltip.style.borderRadius = '4px';
            tooltip.style.fontSize = '14px';
            tooltip.style.zIndex = '1000';
            tooltip.style.whiteSpace = 'nowrap';

            document.body.appendChild(tooltip);

            const rect = el.getBoundingClientRect();
            tooltip.style.top = (rect.top - tooltip.offsetHeight - 10) + 'px';
            tooltip.style.left = (rect.left + (rect.width - tooltip.offsetWidth) / 2) + 'px';

            el.tooltipElement = tooltip;
        });

        el.addEventListener('mouseleave', function() {
            if (el.tooltipElement) {
                el.tooltipElement.remove();
                el.tooltipElement = null;
            }
        });
    });
}

function initCopyButtons() {
    // Add copy functionality to code blocks
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(codeBlock => {
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋';
        copyButton.style.position = 'absolute';
        copyButton.style.top = '8px';
        copyButton.style.right = '8px';
        copyButton.style.background = '#3B82F6';
        copyButton.style.color = 'white';
        copyButton.style.border = 'none';
        copyButton.style.borderRadius = '4px';
        copyButton.style.padding = '4px 8px';
        copyButton.style.cursor = 'pointer';
        copyButton.style.fontSize = '12px';

        const pre = codeBlock.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(copyButton);

        copyButton.addEventListener('click', function() {
            const code = codeBlock.textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyButton.innerHTML = '✓ Copied!';
                copyButton.style.background = '#10B981';

                setTimeout(() => {
                    copyButton.innerHTML = '📋';
                    copyButton.style.background = '#3B82F6';
                }, 2000);
            });
        });
    });
}

function initDiagrams() {
    // Make diagrams interactive
    const diagrams = document.querySelectorAll('.diagram-container');

    diagrams.forEach(diagram => {
        diagram.addEventListener('click', function(e) {
            if (e.target.classList.contains('diagram-node')) {
                const nodeId = e.target.getAttribute('data-node-id');
                showNodeInfo(nodeId);
            }
        });
    });
}

function showNodeInfo(nodeId) {
    // Show information about a diagram node
    const modal = document.createElement('div');
    modal.className = 'diagram-modal';
    modal.style.position = 'fixed';
    modal.style.top = '50%';
    modal.style.left = '50%';
    modal.style.transform = 'translate(-50%, -50%)';
    modal.style.background = 'white';
    modal.style.padding = '20px';
    modal.style.borderRadius = '8px';
    modal.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
    modal.style.zIndex = '1000';

    modal.innerHTML = `
        <h3>Node: ${nodeId}</h3>
        <p>Information about this component...</p>
        <button onclick="this.parentElement.remove()">Close</button>
    `;

    document.body.appendChild(modal);

    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function initProgressAnimations() {
    // Animate progress bars
    const progressBars = document.querySelectorAll('.progress-bar');

    progressBars.forEach(bar => {
        const progress = bar.getAttribute('data-progress');
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = progress + '%';
        }, 100);
    });
}

// Utility functions
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + S to save
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        alert('Save functionality coming soon!');
    }

    // Escape to close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.diagram-modal');
        modals.forEach(modal => modal.remove());
    }
});