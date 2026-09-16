import re


class ResponseValidator:

    def validate(self, response, customer_message):

        issues = []

        if not response or not response.strip():
            issues.append("empty_response")

        response_lower = response.lower()

        # Prevent unsupported claims of account access
        account_claims = [
            "i checked your account",
            "we checked your account",
            "i have checked your account",
            "we have checked your account",
            "i accessed your account",
            "we accessed your account"
        ]

        for claim in account_claims:
            if claim in response_lower:
                issues.append("unsupported_account_access_claim")
                break

        # Detect URLs
        urls = re.findall(
            r"https?://\S+",
            response
        )

        if urls:
            issues.append("contains_url")

        # Basic response length check
        if len(response.strip()) < 10:
            issues.append("response_too_short")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "response": response.strip()
        }