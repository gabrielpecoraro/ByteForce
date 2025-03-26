const doc = window.parent.document;

doc.addEventListener('keydown', function(e) {
    // Enter key for form submission
    if (e.key === 'Enter' && !e.shiftKey) {
        const submitButton = doc.querySelector('button[kind="primary"]');
        if (submitButton) {
            e.preventDefault();
            submitButton.click();
        }
    }
    
    // Ctrl/Cmd + M to toggle mode
    if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        const modeButton = doc.querySelector('button[key="mode_switch"]');
        if (modeButton) {
            modeButton.click();
        }
    }
});

// Add tooltip to show keyboard shortcuts
window.addEventListener('load', function() {
    const modeButton = doc.querySelector('button[key="mode_switch"]');
    if (modeButton) {
        modeButton.setAttribute('title', 'Toggle Mode (⌘/Ctrl + M)');
    }
    
    const submitButton = doc.querySelector('button[kind="primary"]');
    if (submitButton) {
        submitButton.setAttribute('title', 'Submit (Enter)');
    }
});