from typing import List
from urllib.parse import urlparse

import requests
from pydantic.env_settings import BaseSettings
from pydantic.fields import Field


class AzureAdSettings(BaseSettings):
    """
    Represents the Azure AD Settings.
    client_id       : Application Resgistration app id (GUID)
    authority       : Authority url.
        ex: "https://login.microsoftonline.com/{client_id}"
    domain          : Azure tenant domain. "ex:microsoft.onmicrosoft.com"
    tenant_id       : Azure tenant id. (GUID)
    api_scopes      : API scopes list (separated by a blank space).
        ex:"user_impersonation"
    graph_scopes    : GRAPH scopes list (separated by a blank space). ex:"User.Read"
    vault_url : vault url, containing the certificate to use.
        ex:"https://rgbertkvd10.vault.azure.net/"
    vault_certificate_name : certificate name, contained in vault. ex:"mycert"

    if not present, each value will be retrieved from environment variables:
    "CLIENT_ID", "AUTHORITY", "DOMAIN", "TENANT_ID", "API_SCOPES",
    "GRAPH_SCOPES", "VAULT_URL", "VAULT_CERTIFICATE_NAME"

    """

    client_id: str = Field(None, description="Client id", env="CLIENT_ID")
    authority: str = Field(None, description="login authority", env="AUTHORITY")
    domain: str = Field(None, description="Domain name", env="DOMAIN")
    tenant_id: str = Field(None, description="Tenant Id", env="TENANT_ID")
    api_scopes_str: str = Field(None, description="API Scopes", env="API_SCOPES")
    graph_scopes_str: str = Field(None, description="API Scopes", env="GRAPH_SCOPES")
    vault_url: str = Field(None, description="Global Vault Url", env="VAULT_URL")
    vault_certificate_name: str = Field(
        None, description="Certificate name", env="VAULT_CERTIFICATE_NAME"
    )
    aad_issuers_list: List[str] = []

    @property
    def authorization_url(self):
        return f"{self.authority}/oauth2/v2.0/authorize"

    @property
    def token_url(self):
        return f"{self.authority}/oauth2/v2.0/token"

    @property
    def keys_url(self):
        return f"{self.authority}/discovery/v2.0/keys"

    # @property
    # def issuer(self):
    #     return f"https://sts.windows.net/{self.tenant_id}/"

    def get_available_issuers(self):
        if self.aad_issuers_list is not None and len(self.aad_issuers_list) > 0:
            return self.aad_issuers_list

        issuers_list_url = (
            "https://login.microsoftonline.com/common/discovery/instance"
            + "?authorization_endpoint="
            + "https://login.microsoftonline.com/common/oauth2/v2.0/"
            + "authorize&api-version=1.1"
        )

        _issuers_list = requests.get(issuers_list_url).json()
        _metadatas = _issuers_list["metadata"]

        _authority_parser = urlparse(self.authority)
        _authority_domain = _authority_parser.hostname

        for _metadata in _metadatas:
            if _authority_domain in _metadata["preferred_network"]:
                for alias in _metadata["aliases"]:
                    self.aad_issuers_list.append(f"https://{alias}/{self.tenant_id}/")
                    self.aad_issuers_list.append(
                        f"https://{alias}/{self.tenant_id}/v2.0"
                    )

        return self.aad_issuers_list

    @property
    def audiences(self):
        """
        Returns the audiences that are valid for a client_id
        """
        return [
            f"https://{self.domain}/{self.client_id}",
            f"api://{self.client_id}",
            self.client_id,
        ]

    @property
    def graph_scopes(self):
        if self.graph_scopes_str is None:
            return None

        return list(filter(None, self.graph_scopes_str.split(" ")))

    @property
    def api_scopes(self):
        if self.api_scopes_str is None:
            return None

        return list(filter(None, self.api_scopes_str.split(" ")))

    @property
    def scopes_identifiers(self):

        if self.api_scopes is None and self.graph_scopes is None:
            return []

        _api_scopes = []
        _graph_scopes = []

        if self.api_scopes_identifiers is not None:
            _api_scopes = self.api_scopes_identifiers

        if self.graph_scopes is not None:
            _graph_scopes = self.graph_scopes

        # _graph_scopes.extend(_api_scopes)
        # return _graph_scopes
        _api_scopes.extend(_graph_scopes)
        return _api_scopes

    @property
    def api_scopes_identifiers_root(self) -> List[str]:
        if self.api_scopes is None:
            return None

        scopes_identifiers = []
        for ls in self.api_scopes:
            scopes_identifiers.append(f"https://{self.domain}/{self.client_id}")

        return scopes_identifiers

    @property
    def api_scopes_identifiers(self) -> List[str]:
        if self.api_scopes is None:
            return None

        scopes_identifiers = []
        for ls in self.api_scopes:
            scopes_identifiers.append(f"https://{self.domain}/{self.client_id}/{ls}")

        return scopes_identifiers

    @property
    def api_scopes_identifiers_default(self) -> List[str]:
        if self.api_scopes is None:
            return None

        scopes_identifiers = []
        for ls in self.api_scopes:
            scopes_identifiers.append(
                f"https://{self.domain}/{self.client_id}/.default"
            )

        return scopes_identifiers

    @property
    def graph_scopes_identifiers_default(self) -> List[str]:
        return ["https://graph.microsoft.com/.default"]

    class Config:
        env_file = ".env"
