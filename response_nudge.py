"""Briefly favor a duplex reply after audible speech has ended."""


class ResponseNudge:
    def __init__(self, speech_floor_dbfs=-48.0, max_attempts=2, listen_logit_penalty=3.0):
        self.speech_floor_dbfs = speech_floor_dbfs
        self.max_attempts = max_attempts
        self.listen_logit_penalty = listen_logit_penalty
        self.heard_speech = False
        self.attempts = 0

    def before_generate(self, mic_dbfs, logits, listen_token_id):
        """Return the attempt number when the pending listen logit was lowered."""
        if mic_dbfs > self.speech_floor_dbfs:
            self.heard_speech = True
            self.attempts = 0
            return 0
        if not self.heard_speech or self.attempts >= self.max_attempts or logits is None:
            return 0
        self.attempts += 1
        logits[0, listen_token_id] -= self.listen_logit_penalty
        return self.attempts

    def after_generate(self, is_listen):
        exhausted = self.heard_speech and is_listen and self.attempts >= self.max_attempts
        if not is_listen or exhausted:
            self.heard_speech = False
            self.attempts = 0
        return exhausted
