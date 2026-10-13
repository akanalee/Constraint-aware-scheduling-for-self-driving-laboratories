class CommandBot:
    def get_classification(self, text):
        text_lower = text.lower()
        if 'stop' in text_lower:
            return 'stop'
        elif 'restart' in text_lower:
            return 'restart'
        elif 'progress' in text_lower:
            return 'progress_request'
        elif 'hello' in text_lower or 'hi' in text_lower:
            return 'greeting'
        else:
            return 'start'

    def response(self, text):
        return "Acknowledged: processing request for {@EXP_PROCED}"
