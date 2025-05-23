

def save_report_as_markdown(writer_ap,map,output_path,k):
    lines = []
    lines.append(f"| Writer ID | AP@{k} |")
    lines.append("|-----------|-------|")
    for writer_id in sorted(writer_ap):
        lines.append(f"| {int(writer_id):03d} | {writer_ap[writer_id]:.4f} |")

    # save to the file
    with open(output_path, "w") as f:
        f.write("\n")
        f.write("\n")
        f.write("## TOP_K Signature Verification\n")
        f.write("\n")
        f.write("\n".join(lines))
        f.write("\n")
        f.write(f"\n- **Final Mean Average Precision (mAP)**: {map:.4f}")

    print(f"Writer-wise AP@{k} saved to {output_path}")