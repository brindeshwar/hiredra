-- 0006_company_fold: one employer written two ways ("Together AI" from a job board, "Togetherai" from a board slug)
-- has two company keys that differ only by hyphens (decision D2.4, build slice 2.2). A new spelling is matched to an
-- existing company by this squashed form before a new company is created, so the index keeps that lookup cheap.
CREATE INDEX companies_fold ON companies ((replace(slug, '-', '')));
