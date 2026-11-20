News
====

3.5.9
-----

* Cross-file DAP4 views
  * New `pydap.client.open_mfdataset(urls, *, concat_dim, keep_variables,
    overlap='error', session=None, max_buffer_bytes=8388608)` opens a lazy,
    closeable, `DatasetType`-compatible view concatenating multiple DAP4
    granules along a numeric coordinate dimension, without an xarray
    dependency or downloading whole granules.
  * Coordinates are merged in ascending order with exact int64/float64
    comparisons (integers above `2**53` stay distinct) and `error`/`first`/
    `last` policies for cross-file duplicate coordinates; the chosen source
    decides whole rows of every kept variable.
  * Full metadata compatibility checks (dtype, dimension sets/lengths,
    coordinate values, units/calendar/fill/scale/offset, static values),
    axis-order tolerance following the first URL, group-scoped dimensions,
    multi-axis integer/slice/negative-step indexing, and buffer-budgeted
    dynamic payloads.
  * `to_netcdf` accepts a view plus `output_file` to stream one merged
    NetCDF4 file with atomic replacement; existing single/multi-URL styles
    are unchanged.
  * DAP4 per-variable CRC32 checksums are now verified and truncated
    responses raise instead of returning partial arrays.
  * `consolidate_metadata(..., safe_mode=True)` now compares shared
    coordinate *values*, not only dimension lengths; `http://` sources are
    accepted for local testing.
* Server-side grouped aggregation
  * New server function `group_reduce(sequence, keys_json, aggregates_json)`
    groups a flat DAP2 `Sequence` by key columns and computes `count`,
    `sum`, `mean`, `min` and `max` on the server, returning a `summary`
    `Sequence` (`Int32` counts, `Float64` other aggregates). Input is
    traversed once; simultaneously resident group states are bounded by the
    new `ServerSideFunctions` options `group_reduce_max_groups=1024` and
    `group_reduce_tempdir=None`, with sorted spill files and batched merge
    passes for near-all-distinct keys.
  * Lazy, order-preserving client views add `where(expression)` (comparisons
    and AND over the grouped result), `select(*fields)` (projection) and
    `limit(n, offset=0)` (row window). The stages are encoded as replayable
    DAP2 function-call constraint expressions and executed entirely on the
    server; view creation sends no request.
  * DAP2 string transport now uses UTF-8 so non-ASCII string keys and
    constants round-trip through DDS/DODS responses.

3.4.0
-----

*Release date: 2023-April-5*

* DAP4/DAS Patch (#278)
  * Fixing a type that was causing DAP4 content to be igested in a DAP2 object with bad results.
  * Retiring tests for Python 3.6 as it is no longer available in ubuntu. Adding tests for Python 3.10 and 3.11
  * Dropped 3.7 and 3.8 from the version matrix
  * test_response_error regex fixed
  * test_response_error fix for 3.10 and 3.9
  * changed python_macro version to 3.9
  * Updated .gitignore for OSX Finder stuff

* Dap4 beta (#271)
  * Initial DAP4 support
  * These changes are useful but do not represent 100% conformance to the DAP4.

3.3.0
-----

*Release date: 2022-Feb-1*

* There are many changes going from 3.2.0 to 3.3.0

  Merge pull request #259 from pydap/ejh_readme
  fixing documentation links in README

  Merge pull request #258 from pydap/ejh_version
  changed version to 3.3.0

  Merge pull request #253 from pydap/ejh_remove_python_2
  removing support for python 2

  Merge pull request #209 from shreddd/master
  Fix to speed up directory listings.

  Merge pull request #257 from pydap/ejh_macos_2
  adding macos to GitHub CI

  Merge branch 'timeout' of github.com:cskarby/pydap
  Merge branch 'float_inf' of github.com:d70-t/pydap into ejh_inf

  Merge pull request #247 from pydap/ejh_warn2
  fixed tostring warnings

  Merge pull request #246 from pydap/ejh_warn
  fixed pytest warnings

  Merge pull request #243 from pydap/ejh_collections
  Import ABC from collections.abc instead of collections for Python 3 compatibility
  adding more versions of python

  Merge pull request #241 from pydap/ejh_t1
  adding GitHub actions CI workflow

  fix pos arg (#225)

  WIP: add user_charset arg (#223)

  Add 'default_charset' to open_url, for servers that don't (#222)
  specify it but serve utf-8

  client: Bugfix - pass session to server functions
  self.session was set to None instead of passed session object

  client: Pass timeout from open_url to server functions as well
  Fixes pydap/pydap#220

  Handle KeyErrors as described in issue #128 (#201)

  Do not yield faulty DAS lines (#195)
  Fix #194.
  `sudo` is no longer needed (#193)
  https://blog.travis-ci.com/2018-11-19-required-linux-infrastructure-migration

  Avoid applying scale factor for consistency with dtype (#191)
  Fix #190.
  Fix Travis CI (#192)
  Fix PEP 479
  Fix PEP8
  add netcdf handler entry point (#179)
  minor client doc fixes (#181)
  adding logo pydap (#178)

  Merge pull request #177 from tomkralidis/add-tomkralidis-to-contributors

  Explicitly test client-only installations (solves #120) (#124)
  Add a ``verify`` option to ``open_url`` (#112)

  Add a netcdf4-python interface for testing (#106)
  Remove serverside functions in devel server (#123)
  Remove non-existent entry in index (#115)
  make pydap canonical project name (#174)
  simplify HTML (#144)
  Fix gzip compression (#126) (#152)
  add support for THREDDS Catalog XML (#136) (#138)
  add support for THREDDS Catalog XML (#136)
  add pydap.__version__ capability (#133) (#135)

  Merge pull request #172 from rsignell-usgs/patch-1
  Create CODE_OF_CONDUCT.md

  Merge pull request #173 from betodealmeida/fix_ci_35
  Fix unit tests in Python 3.5

  Create CODE_OF_CONDUCT.md

  Merge pull request #161 from flackdl/patch-1
  Fix bad link in README

  add LICENSE file (#142) (#143)

  Handle gzipped responses (Solves #126) (#127)
  add requests to core dependencies (#145) (#120) (#146)
  Fixes #121, fixes deprecation warnings and incorporates flake8 changes (#159)
  Fix title level inconsistency (#117)
  Closes #116
  Exclude the Sphinx build directory from sdist (#114)
  Do not require mock for Python 3.3 or later (#113)
  `unittest.mock` is available since Python 3.3

  Merge pull request #34 from laliberte/merging/handlers.csv
  Merges CSV handler into the main repo

  Merge branch 'release/3.2.2'
  Add PasteDeploy as an optional dependency. Fixes #53
  New way of running a Paste server with Gunicorn. Fixes #52

3.2.2
-----

*Release date: 2017-May-24*

* Python 3.3 is no longer supported. This is in line with
  other similar projects (Numpy, Xarray, ...) and it preempts the
  expected python 3.3 EOL in September 2017.
* Server improvements
  * Merges pydap.handlers.netcdf into the main code base
  * Adds a lightweight testing/development server
  * Rewrites the server docs to reflect a post-paster world
* Miscellaneous bug fixes
  * Ensures Byte use is consistent with DAP2 standards
  * Fixes client authentication to UK's CEDA
  * Fixes client communcation with ERDDAP servers
  * Fixes regression bug in model.GridType (#43)
  * Fixes bug where the iteration does not replace previous_chunk
  * Fixes bugs in command-line server (#52 and #53)
* Fix mapping scheme for SequenceType (PR #89)
  * Makes all types a mapping and protects the mapping semantics for sequence data
  * Converts dict in StructureType to OrderedDict, changing
    _original_keys handling (#3) and convers test_model.py to pytest
    semantics (#82).
  * Updats docs and docstrings. Added basic automatic
    doctests. Doctests use integers for easy continuous integration
  * 100% test coverage in src/pydap/model.py.
* Various codebase improvements
  * Transition tests from nose to pytest
  * Tests with flake8 on all Python version
* Adds timeout option to open_urls and open_dods


3.2.1
-----

*Release date: 2017-Mar-28*

* PyDAP client fixes
  * Adds workaround to the client when making requests to older ESGF OPENDAP servers
  * Fixes a bug where mechanicalsoup wouldn't its browser
  * Adds handling for values of -NaN
  * Overhauls URS authentication to use the requests library
  * Sets a default charset when connecting to servers that do not
* Packaging fixes
  * Moves gunicorn in to server_extras dependency list
  * Adds test data to the release tarball
* Code base improvements
  * Adds flake8 linting/checking to the code base
  * Improves testing for client authentication
  * Drops support for Python < 2.7
  * Converts internal imports to be explicit relative
* Miscellaneous bug fixes
  * Fixes bug in fix_slices() when `stop` > len
  * Fixes unexpected flattening of sliced arrays (#41)
  * Fixes bug where attributes were not propagated to nested structures (#75)


3.2.0
-----

*Release date: 2016-Dec-01*

* Adds some optimizations to the server for DAP sequences
* Rewrite of the client so that it's now able to stream real time data
* Simplifies the design of handlers, eliminating DAP-related details
  (so that developers can focus only on the data parsing when creating
  new handlers).
* Full test coverage and continuous integration
* Adds support for Python version 3.3 through 3.5
* Adds support for federated authentication through the Earth System
  Grid Federation (ESGF) and NASA User Registration System (URS).
* Fixes HTML response to show all dimensions for BaseType variables

3.1.1
-----

*Release date: 14-Nov-2013*

* Final release done by Roberto De Almeida
