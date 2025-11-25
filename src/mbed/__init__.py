import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Main entry point for mbed CLI."""
    parser = argparse.ArgumentParser(
        description="mbed - Minimal Embeddings with Vector Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize index for a directory")
    init_parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Directory to index (default: current directory)",
    )
    init_parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model to use",
    )
    init_parser.add_argument(
        "--storage",
        choices=["chromadb"],
        default="chromadb",
        help="Vector storage backend",
    )
    init_parser.add_argument(
        "--top-k", type=int, default=3, help="Default number of search results"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed directory")
    search_parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Indexed directory (default: current directory)",
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=None, help="Override number of results"
    )

    # Update command
    update_parser = subparsers.add_parser(
        "update", help="Update index with new/modified files"
    )
    update_parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Indexed directory (default: current directory)",
    )
    update_parser.add_argument(
        "-y", "--yes", action="store_true", help="Auto-confirm changes"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Check for file changes")
    status_parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Indexed directory (default: current directory)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle commands
    try:
        if args.command == "init":
            from .indexer import create_index

            create_index(
                args.directory, args.model, args.storage, top_k=args.top_k
            )
            print(f"Index created at {args.directory / '.mbed'}")

        elif args.command == "search":
            from .searcher import search_directory

            response = search_directory(args.directory, args.query, args.top_k)
            print(response)

        elif args.command == "update":
            from .file_tracker import detect_changes, update_index

            # First detect changes
            changes = detect_changes(args.directory)
            total_changes = (
                len(changes["added"])
                + len(changes["modified"])
                + len(changes["deleted"])
            )

            if total_changes == 0:
                print("No changes detected. Index is up to date.")
                return

            # Display changes
            print("Detected changes:")
            if changes["added"]:
                print(f"  Added: {len(changes['added'])} files")
                for change in changes["added"][:5]:
                    print(f"    + {change.path.name}")
                if len(changes["added"]) > 5:
                    print(f"    ... and {len(changes['added']) - 5} more")

            if changes["modified"]:
                print(f"  Modified: {len(changes['modified'])} files")
                for change in changes["modified"][:5]:
                    print(f"    ~ {change.path.name}")
                if len(changes["modified"]) > 5:
                    print(f"    ... and {len(changes['modified']) - 5} more")

            if changes["deleted"]:
                print(f"  Deleted: {len(changes['deleted'])} files")
                for change in changes["deleted"][:5]:
                    print(f"    - {change.path.name}")
                if len(changes["deleted"]) > 5:
                    print(f"    ... and {len(changes['deleted']) - 5} more")

            # Confirm if not auto-confirmed
            if not args.yes:
                response = input("\nApply changes? [y/N]: ")
                if response.lower() != "y":
                    print("Cancelled.")
                    return

            # Apply updates
            result = update_index(args.directory)

            # Display results
            if result["errors"]:
                print(f"\nProcessed {result['processed']} files with {len(result['errors'])} errors:")
                for file_path, error in result["errors"]:
                    print(f"  Error: {file_path.name} - {error}")
            else:
                print(f"\nIndex updated successfully! Processed {result['processed']} files.")

            if changes["deleted"]:
                deleted_count = len(changes["deleted"])
                print(f"\nRemoved {deleted_count} deleted file(s) from index.")

        elif args.command == "status":
            from .file_tracker import detect_changes

            changes = detect_changes(args.directory)
            total = (
                len(changes["added"])
                + len(changes["modified"])
                + len(changes["deleted"])
            )

            if total == 0:
                print("No changes detected. Index is up to date.")
            else:
                print("File changes detected:")
                print(f"  Added:    {len(changes['added'])} files")
                print(f"  Modified: {len(changes['modified'])} files")
                print(f"  Deleted:  {len(changes['deleted'])} files")
                print("\nRun 'mbed update' to apply changes.")

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
