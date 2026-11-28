# Files API: the store directory

`--files-dir <dir>` turns the Files API on (`/v1/files`). This page is for the operator: what the
directory holds, how the quota is kept, and how to remove files the API can no longer reach.

## What the directory holds

- `<id>.data`: a file's bytes.
- `<id>.json`: its metadata (`id`, `filename`, `bytes`, `created_at`, `purpose`, `mime_type`,
  `downloadable`, `owner`). A file exists once this is written.
- `next`: the id counter, so the id of a deleted file is never handed out again.
- `.staging-*`: uploads in progress. They are removed at start.

Ids are `file-<12 hex digits of the counter><12 random hex digits>`.

## The quota

- `--files-quota-gb` is the total the files may hold. Uploads in progress reserve their room
  before their bytes are read: the declared body size, or the per-file cap (`--files-max-mb`) for
  a chunked upload, at most the free bytes. The status line and the refusals count those
  reservations as in use. An upload larger than what it could reserve is refused as soon as it
  passes it.
- At start the files in the directory are counted. If they hold more than the quota (the quota
  was lowered since), the server does not start and says how many bytes they hold. Raise the
  quota, or stop the server and remove files as below.

## Owners

- With `--api-key`, a file belongs to the key that stored it. `owner` in its `.json` is
  `key:` and the first 32 hex digits of the key's SHA-256 digest. Without a key, `owner` is
  empty, and every client sees every file stored without a key.
- The owner is the server's key that let the request in, as a Bearer token or as `x-api-key`. A
  request may carry both; when one of them is the key, the other is ignored, and it never picks
  the owner. A request with neither is refused (401) before any file is looked at.
- After the key changes (or one is added to a server that had none, or removed), the files
  stored under the old owner are not served: list, get, content and delete answer as if they did
  not exist. Their bytes still count against the quota. The start reports them:
  `[server] Files API: <n> files (<bytes> bytes) are not served ...`. Files whose `.json` cannot
  be read are counted in the same line.

## Removing files by hand

Stop the server first: the directory is read only at start.

The owner digest of a key, and the files stored under any other owner:

```sh
key_owner="key:$(printf %s "$KEY" | sha256sum | cut -c1-32)"   # or key_owner="" with no key
cd <files-dir>
for j in file-*.json; do
  grep -q "\"owner\":\"$key_owner\"" "$j" || echo "${j%.json}"
done
```

To remove a file, delete both `<id>.json` and `<id>.data`. Leave `next` in place so that the
removed ids stay retired. A `.data` without its `.json` is removed at the next start; a `.json`
without its `.data` too.
