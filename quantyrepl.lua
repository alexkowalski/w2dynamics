function main()
   io.stdout:flush()
   io.stderr:flush()

   while true do

      local instr = ""
      while true do
	 local nextline = io.stdin:read("*L")
	 if nextline == nil then
	    do return end
	 elseif nextline == "EOF\n" then
	    break
	 else
	    instr = instr .. nextline
	 end
      end

      local fn = load(instr, instr, "t")
      instr = ""
      a, b = pcall(fn)
      if not a then
	 io.stderr:write(b)
      end
      io.stdout:write("\nEOF\n")
      io.stdout:flush()
      io.stderr:write("\nEOF\n")
      io.stderr:flush()
      collectgarbage()

   end
end

main()
