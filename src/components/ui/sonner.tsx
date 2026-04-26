import { Toaster as Sonner } from "sonner";

type ToasterProps = React.ComponentProps<typeof Sonner>;

const Toaster = ({ ...props }: ToasterProps) => {
	return (
		<Sonner
			className="toaster group"
			duration={3000}
			toastOptions={{
				classNames: {
					toast: "group toast group-[.toaster]:bg-[#1C1C1C] group-[.toaster]:text-[#EDEDED] group-[.toaster]:border-white/[0.06] group-[.toaster]:shadow-lg",
					description: "group-[.toast]:text-muted-foreground",
					actionButton: "group-[.toast]:bg-[#D4D0C8] group-[.toast]:text-[#0A0A0A]",
					cancelButton: "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground",
				},
			}}
			{...props}
		/>
	);
};

export { Toaster };
