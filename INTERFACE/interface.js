const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        const submitButton = doc.querySelector('button[kind="primary"]');
        if (submitButton) {
            e.preventDefault();
            submitButton.click();
        }
    }
});