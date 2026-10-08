from dataclasses import dataclass

from sqlalchemy.orm import Session

from finsight.clients.sec import SecClient, SecFiling
from finsight.db.models import Filing
from finsight.repositories.filing import FilingRepository


@dataclass(frozen=True)
class IngestionResult:
    ticker: str
    discovered: int
    created: int
    skipped: int


class FilingIngestionService:
    def __init__(
        self,
        session: Session,
        sec_client: SecClient,
    ) -> None:
        self._session = session
        self._sec_client = sec_client
        self._repository = FilingRepository(session)

    async def ingest(
        self,
        ticker: str,
        form_types: tuple[str, ...] = ("10-K", "10-Q"),
        limit: int = 10,
    ) -> IngestionResult:

        company = await self._sec_client.get_company(ticker)
        sec_filings = await self._sec_client.get_recent_filings(
            company,
            form_types=form_types,
            limit=limit,
        )

        created = 0
        skipped = 0

        try:
            for sec_filing in sec_filings:
                # find if it already exists
                existing = self._repository.get_by_accession_number(sec_filing.accession_number)

                if existing is not None:
                    skipped += 1
                    continue

                filing = self._to_model(sec_filing)
                self._repository.add(filing)
                created += 1

            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        return IngestionResult(
            ticker=company.ticker,
            discovered=len(sec_filings),
            created=created,
            skipped=skipped,
        )

    @staticmethod
    def _to_model(sec_filing: SecFiling) -> Filing:
        return Filing(
            accession_number=sec_filing.accession_number,
            cik=sec_filing.cik,
            company_name=sec_filing.company_name,
            form_type=sec_filing.form_type,
            filing_date=sec_filing.filing_date,
            report_period=sec_filing.report_period,
            source_url=sec_filing.source_url,
            status="discovered",
        )
